import os
import json
import multiprocessing
from multiprocessing.pool import Pool
from pathlib import Path
from typing import Iterator, Sequence

import lightgbm as lgb
import numpy as np
import polars as pl
import tqdm
from sklearn.metrics import make_scorer, roc_auc_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, train_test_split

from .features import PEFeatureExtractor


FEATURE_NAMES_FILENAME = "feature_names.json"
METADATA_FILENAME = "metadata.parquet"
DATASET_SUBSETS = ("train", "test", "challenge")
PARQUET_BATCH_SIZE = 1024

ORDERED_COLUMNS = [
    "sha",
    "sha256",
    "timestamp",
    "subset",
    "tlsh",
    "first_submission_date",
    "last_analysis_date",
    "detection_ratio",
    "label",
    "file_type",
    "family",
    "family_confidence",
    "behavior",
    "file_property",
    "packer",
    "exploit",
    "group",
]

LIST_METADATA_COLUMNS = {
    "behavior",
    "file_property",
    "packer",
    "exploit",
    "group",
}
STRING_METADATA_COLUMNS = {
    "sha",
    "sha256",
    "subset",
    "tlsh",
    "first_submission_date",
    "last_analysis_date",
    "detection_ratio",
    "file_type",
    "family",
}
INTEGER_METADATA_COLUMNS = {
    "timestamp",
    "label",
}
FLOAT_METADATA_COLUMNS = {
    "family_confidence",
}


def raw_feature_iterator(file_paths: list[Path]) -> Iterator[str]:
    """
    Yield raw feature strings from the inputed file paths
    """
    for path in file_paths:
        with path.open("r") as fin:
            for line in fin:
                yield line


def get_feature_names(features_file: Path | str | None = None) -> list[str]:
    """
    Return all vectorized feature names in the same order as PEFeatureExtractor output.
    """
    features_path = Path(features_file) if features_file is not None else None
    extractor = PEFeatureExtractor(features_path)
    return list(extractor.feature_names)


def write_feature_names(
    data_dir: Path | str,
    output_path: Path | str | None = None,
    features_file: Path | str | None = None,
) -> list[str]:
    """
    Write vectorized feature names to disk and return them.
    """
    data_path = Path(data_dir)
    names = get_feature_names(features_file)
    feature_names_path = Path(output_path) if output_path is not None else data_path / FEATURE_NAMES_FILENAME
    feature_names_path.parent.mkdir(parents=True, exist_ok=True)
    feature_names_path.write_text(json.dumps(names, indent=2) + "\n", encoding="utf8")
    return names


def import_pyarrow():
    """
    Import pyarrow lazily so normal feature extraction does not require it.
    """
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as error:
        raise ImportError(
            "Writing feature parquet files requires pyarrow. "
            "Install project dependencies or run `pip install pyarrow`."
        ) from error
    return pa, pq


def count_rows(file_paths: list[Path]) -> int:
    """
    Count rows in raw feature files.
    """
    nrows = 0
    for file_path in file_paths:
        with file_path.open() as fp:
            nrows += sum(1 for _ in fp)
    return nrows


def gather_feature_paths(data_dir: Path | str, subset: str, filetype: str = None, week: str = None) -> list[Path]:
    """
    Gather paths to raw metadata .jsonl files in the given data_dir
    Supports filtering by train/test/challenge subset, file type, and/or data collection week
    """
    feature_paths = []
    for file_name in sorted(os.listdir(data_dir)):
        if not file_name.endswith(".jsonl"):
            continue
        if subset not in file_name:
            continue
        if filetype is not None and filetype not in file_name:
            continue
        if week is not None and week not in file_name:
            continue
        feature_paths.append(Path(os.path.join(data_dir, file_name)))

    if not len(feature_paths):
        raise ValueError("Did not find any .jsonl files matching criteria")
    return feature_paths



def read_label(raw_features_string: str, label_type: str) -> str:
    """
    Read the label or tag from raw features and return it
    """
    raw_features = json.loads(raw_features_string)
    label = raw_features[label_type]
    return label


def read_label_unpack(args):
    """
    Pass through function for unpacking read_label arguments
    """
    return read_label(*args)


def read_label_subset(raw_feature_paths: list[Path], nrows: int, label_type: str) -> set:
    """
    Read the unique labels/tags in the subset
    """
    # Distribute the vectorization work
    pool = multiprocessing.Pool()
    argument_iterator = (
        (raw_features_string, label_type)
        for _, raw_features_string in enumerate(raw_feature_iterator(raw_feature_paths))
    )
    label_counts = {}
    for labels in tqdm.tqdm(pool.imap_unordered(read_label_unpack, argument_iterator), total=nrows):
        if not isinstance(labels, list):
            labels = [labels]
        for label in labels:
            if label_counts.get(label) is None:
                label_counts[label] = 0
            label_counts[label] += 1
    return label_counts


def vectorize(irow: int, raw_features_string: str, X_path: str, y_path: str, extractor: PEFeatureExtractor, nrows: int, label_type: str = "label", label_map: dict = {}) -> None:
    """
    Vectorize a single sample of raw features and write to a large numpy file
    """
    raw_features = json.loads(raw_features_string)
    feature_vector = extractor.process_raw_features(raw_features)

    if label_type not in raw_features:
        raise ValueError("Invalid label_type!")
    label = raw_features[label_type]

    # Figure out what 'label' is
    if label is None and (label_type == "label" or label_type == "family"):
        y = np.memmap(y_path, dtype=np.int32, mode="r+", shape=nrows)
        y[irow] = -1
    elif isinstance(label, int): # Benign/Malicious labels (binary)
        y = np.memmap(y_path, dtype=np.int32, mode="r+", shape=nrows)
        y[irow] = label
    elif isinstance(label, str): # Family labels (multiclass)
        y = np.memmap(y_path, dtype=np.int32, mode="r+", shape=nrows)
        if label_map.get(label) is not None:
            y[irow] = label_map[label]
        else:
            y[irow] = -1
    elif isinstance(label, list): # Tags (multiclass, multilabel)
        y = np.memmap(y_path, dtype=np.int32, mode="r+", shape=(nrows, len(label_map.keys())))
        for l in label:
            if label_map.get(l) is not None:
                y[irow,label_map[l]] = 1
    else:
        raise ValueError("Unable to parse label format")

    X = np.memmap(X_path, dtype=np.float32, mode="r+", shape=(nrows, extractor.dim))
    X[irow] = feature_vector


def vectorize_unpack(args):
    """
    Pass through function for unpacking vectorize arguments
    """
    return vectorize(*args)


def vectorize_subset(X_path: Path, y_path: Path, raw_feature_paths: list[Path], extractor: PEFeatureExtractor, nrows: int, label_type: str = "label", label_map: dict = {}) -> None:
    """
    Vectorize a subset of data and write it to disk
    """
    # Create space on disk to write features to
    X = np.memmap(X_path, dtype=np.float32, mode="w+", shape=(nrows, extractor.dim))
    if label_type == "label" or label_type == "family":
        y = np.memmap(y_path, dtype=np.float32, mode="w+", shape=nrows)
    else:
        y = np.memmap(y_path, dtype=np.float32, mode="w+", shape=(nrows, len(label_map.keys())))
    del X, y

    # Distribute the vectorization work
    pool = multiprocessing.Pool()
    argument_iterator = (
        (irow, raw_features_string, X_path, y_path, extractor, nrows, label_type, label_map)
        for irow, raw_features_string in enumerate(raw_feature_iterator(raw_feature_paths))
    )
    for _ in tqdm.tqdm(pool.imap_unordered(vectorize_unpack, argument_iterator), total=nrows):
        pass


def create_vectorized_features(
    data_dir: Path | str,
    label_type: str = "label",
    class_min: int = 10,
    create_parquet: bool = False,
    parquet_output_dir: Path | str | None = None,
    parquet_batch_size: int = PARQUET_BATCH_SIZE,
) -> None:
    """
    Create feature vectors from raw features and write them to disk

    Arguments:
    data_dir - Path to the directory containing the dataset.
    label_type - The type of classification problem.
    class_min - The minimum number of instances of a class in the dataset. Data
                points belonging to a class with fewer than class_min instances
                are ignored.
    create_parquet - Whether to also create train/test/challenge parquet files
                     with metadata and named feature columns.
    parquet_output_dir - Optional directory for parquet outputs.
    parquet_batch_size - Number of rows per parquet write batch.

    Valid label_types:
    label - malicious/benign (binary)
    family - malware family classification (multiclass)
    behavior - malware behavior prediction (multiclass, multi-label)
    file_property - malware file property prediction (multiclass, multi-label)
    packer - malware packer prediction (multiclass, multi-label)
    exploit - malware exploit prediction (multiclass, multi-label)
    group - malware threat group prediction (multiclass, multi-label)
    """
    # Ignore empty tags and self-describing file format tags
    ignore_tags = set(["", "win32", "win64", "elf", "linux", "pdf", "apk", "android"])

    extractor = PEFeatureExtractor()
    data_path: Path = Path(data_dir)
    write_feature_names(data_path)

    print("Preparing to vectorize raw features")
    X_train_path = data_path / "X_train.dat"
    y_train_path = data_path / "y_train.dat"
    train_feature_paths = gather_feature_paths(data_path, "train")
    train_nrows = count_rows(train_feature_paths)

    X_test_path = data_path / "X_test.dat"
    y_test_path = data_path / "y_test.dat"
    test_feature_paths = gather_feature_paths(data_path, "test")
    test_nrows = count_rows(test_feature_paths)

    # Map string labels/tags to numeric labels
    label_map = {}
    i = 0
    if label_type != "label": # No work needed for the default malicious/benign labels
        train_label_counts = read_label_subset(train_feature_paths, train_nrows, label_type)

        # Remove labels/tags that appear fewer than class_min time
        for l, count in train_label_counts.items():
            if l in ignore_tags:
                continue
            if count >= class_min:
                label_map[l] = i
                i += 1

    print("Vectorizing training set")
    vectorize_subset(X_train_path, y_train_path, train_feature_paths, extractor, train_nrows, label_type, label_map)

    if label_type != "label": # No work needed for the default malicious/benign labels
        test_label_counts = read_label_subset(test_feature_paths, test_nrows, label_type)

        # Remove labels/tags that appear fewer than class_min time
        for l, count in test_label_counts.items():
            if l in ignore_tags:
                continue
            if label_map.get(l) is not None:
                continue
            if count >= class_min:
                label_map[l] = i
                i += 1

    print("Vectorizing test set")
    vectorize_subset(X_test_path, y_test_path, test_feature_paths, extractor, test_nrows, label_type, label_map)

    print("Vectorizing challenge set")
    X_test_path = data_path / "X_challenge.dat"
    y_test_path = data_path / "y_challenge.dat"
    raw_feature_paths = gather_feature_paths(data_path, "challenge")
    nrows = count_rows(raw_feature_paths)
    vectorize_subset(X_test_path, y_test_path, raw_feature_paths, extractor, nrows)

    if create_parquet:
        print("Writing parquet feature files")
        create_parquet_features(
            data_path,
            output_dir=parquet_output_dir,
            batch_size=parquet_batch_size,
            use_vectorized=True,
        )



def read_vectorized_features(data_dir: Path | str, subset: str = "train") -> tuple[np.ndarray, np.ndarray]:
    """
    Read vectorized features into memory mapped numpy arrays
    """
    data_path: Path = Path(data_dir)
    X_path = data_path / f"X_{subset}.dat"
    y_path = data_path / f"y_{subset}.dat"

    if not os.path.isfile(X_path):
        raise ValueError(f"Invalid subset file: {X_path}")
    if not os.path.isfile(y_path):
        raise ValueError(f"Invalid subset file: {y_path}")

    extractor = PEFeatureExtractor()
    ndim: int = extractor.dim
    X = np.memmap(X_path, dtype=np.float32, mode="r")
    X = np.array(X).reshape(-1, ndim)
    N: int = X.shape[0]
    y = np.memmap(y_path, dtype=np.int32, mode="r")
    y = np.array(y)
    if y.shape[0] > N:
        y = y.reshape(N, -1)

    return X, y


def extract_metadata(raw_features: dict, subset: str | None = None) -> dict:
    """
    Extract metadata fields from a raw feature record.
    """
    header = raw_features.get("header") or {}
    coff = header.get("coff") or {}

    record = {
        key: raw_features.get(key)
        for key in ORDERED_COLUMNS
        if key not in {"sha", "timestamp", "subset"}
    }
    record["sha"] = raw_features.get("sha") or raw_features.get("sha256")
    record["timestamp"] = raw_features.get("timestamp")
    if record["timestamp"] is None:
        record["timestamp"] = coff.get("timestamp")
    if subset is not None:
        record["subset"] = subset
    return record


def read_metadata_record(raw_features_string: str) -> dict:
    """
    Decode a raw features string and return the metadata fields.
    """
    return extract_metadata(json.loads(raw_features_string))


def normalize_metadata_record(record: dict) -> dict:
    """
    Normalize metadata values to stable parquet column types.
    """
    normalized = {}
    for key in ORDERED_COLUMNS:
        value = record.get(key)
        if key in LIST_METADATA_COLUMNS:
            if value is None:
                normalized[key] = []
            elif isinstance(value, list):
                normalized[key] = [str(item) for item in value if item is not None]
            else:
                normalized[key] = [str(value)]
        elif key in STRING_METADATA_COLUMNS:
            normalized[key] = None if value is None else str(value)
        elif key in INTEGER_METADATA_COLUMNS:
            try:
                normalized[key] = None if value is None else int(value)
            except (TypeError, ValueError):
                normalized[key] = None
        elif key in FLOAT_METADATA_COLUMNS:
            try:
                normalized[key] = None if value is None else float(value)
            except (TypeError, ValueError):
                normalized[key] = None
        else:
            normalized[key] = value
    return normalized


def parquet_schema(feature_names: list[str], pa):
    """
    Build the schema for subset parquet files.
    """
    fields = []
    for key in ORDERED_COLUMNS:
        if key in LIST_METADATA_COLUMNS:
            fields.append(pa.field(key, pa.list_(pa.string())))
        elif key in STRING_METADATA_COLUMNS:
            fields.append(pa.field(key, pa.string()))
        elif key in INTEGER_METADATA_COLUMNS:
            fields.append(pa.field(key, pa.int64()))
        elif key in FLOAT_METADATA_COLUMNS:
            fields.append(pa.field(key, pa.float64()))
        else:
            fields.append(pa.field(key, pa.string()))

    fields.extend(pa.field(name, pa.float32()) for name in feature_names)
    metadata = {
        b"thrember_feature_names": json.dumps(feature_names).encode("utf8"),
        b"thrember_metadata_columns": json.dumps(ORDERED_COLUMNS).encode("utf8"),
    }
    return pa.schema(fields, metadata=metadata)


def parquet_batch_to_table(metadata_records: list[dict], feature_matrix: np.ndarray, feature_names: list[str], schema, pa):
    """
    Convert one metadata/feature batch into an Arrow table.
    """
    if len(metadata_records) != feature_matrix.shape[0]:
        raise ValueError("Metadata and feature batch sizes do not match")
    if len(feature_names) != feature_matrix.shape[1]:
        raise ValueError("Feature names do not match feature matrix width")

    batch = {
        key: [record.get(key) for record in metadata_records]
        for key in ORDERED_COLUMNS
    }
    for idx, name in enumerate(feature_names):
        batch[name] = feature_matrix[:, idx]
    return pa.table(batch, schema=schema)


def validate_subset(subset: str) -> None:
    """
    Validate a dataset subset name.
    """
    if subset not in DATASET_SUBSETS:
        raise ValueError(f"Invalid subset: {subset}. Expected one of {DATASET_SUBSETS}")


def write_subset_parquet(
    data_dir: Path | str,
    subset: str,
    output_path: Path | str | None = None,
    batch_size: int = PARQUET_BATCH_SIZE,
    compression: str = "zstd",
    use_vectorized: bool = True,
) -> Path:
    """
    Write one dataset subset as parquet with metadata columns and named feature columns.

    If X_<subset>.dat exists and use_vectorized is true, features are read from
    the vectorized memmap. Otherwise, features are generated from raw JSONL rows.
    """
    validate_subset(subset)
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    pa, pq = import_pyarrow()
    data_path = Path(data_dir)
    parquet_path = Path(output_path) if output_path is not None else data_path / f"{subset}.parquet"
    parquet_path.parent.mkdir(parents=True, exist_ok=True)

    extractor = PEFeatureExtractor()
    feature_names = list(extractor.feature_names)
    feature_paths = gather_feature_paths(data_path, subset)
    nrows = count_rows(feature_paths)
    schema = parquet_schema(feature_names, pa)

    X = None
    X_path = data_path / f"X_{subset}.dat"
    if use_vectorized and X_path.exists():
        expected_size = nrows * extractor.dim * np.dtype(np.float32).itemsize
        actual_size = X_path.stat().st_size
        if actual_size != expected_size:
            raise ValueError(
                f"Unexpected size for {X_path}: expected {expected_size} bytes, found {actual_size}"
            )
        X = np.memmap(X_path, dtype=np.float32, mode="r", shape=(nrows, extractor.dim))

    writer = pq.ParquetWriter(str(parquet_path), schema=schema, compression=compression)
    start = 0
    metadata_records = []
    feature_vectors = []
    try:
        for raw_features_string in tqdm.tqdm(
            raw_feature_iterator(feature_paths),
            total=nrows,
            desc=f"Writing {subset} parquet",
        ):
            raw_features = json.loads(raw_features_string)
            metadata_records.append(
                normalize_metadata_record(extract_metadata(raw_features, subset=subset))
            )
            if X is None:
                feature_vectors.append(extractor.process_raw_features(raw_features))

            if len(metadata_records) == batch_size:
                if X is not None:
                    feature_matrix = np.asarray(X[start:start + len(metadata_records)], dtype=np.float32)
                else:
                    feature_matrix = np.vstack(feature_vectors).astype(np.float32, copy=False)
                writer.write_table(parquet_batch_to_table(metadata_records, feature_matrix, feature_names, schema, pa))
                start += len(metadata_records)
                metadata_records = []
                feature_vectors = []

        if metadata_records:
            if X is not None:
                feature_matrix = np.asarray(X[start:start + len(metadata_records)], dtype=np.float32)
            else:
                feature_matrix = np.vstack(feature_vectors).astype(np.float32, copy=False)
            writer.write_table(parquet_batch_to_table(metadata_records, feature_matrix, feature_names, schema, pa))
    finally:
        writer.close()

    return parquet_path


def create_parquet_features(
    data_dir: Path | str,
    subsets: Sequence[str] = DATASET_SUBSETS,
    output_dir: Path | str | None = None,
    batch_size: int = PARQUET_BATCH_SIZE,
    compression: str = "zstd",
    use_vectorized: bool = True,
) -> dict[str, Path]:
    """
    Write train/test/challenge feature parquet files with metadata and feature names.
    """
    data_path = Path(data_dir)
    output_path = Path(output_dir) if output_dir is not None else data_path
    output_path.mkdir(parents=True, exist_ok=True)
    write_feature_names(data_path, output_path=output_path / FEATURE_NAMES_FILENAME)

    parquet_paths = {}
    for subset in subsets:
        validate_subset(subset)
        parquet_paths[subset] = write_subset_parquet(
            data_path,
            subset,
            output_path=output_path / f"{subset}.parquet",
            batch_size=batch_size,
            compression=compression,
            use_vectorized=use_vectorized,
        )
    return parquet_paths


def read_metadata_subset(data_path: Path, subset: str, pool: Pool) -> pl.DataFrame:
    """
    Read metadata for one raw-feature subset.
    """
    feature_paths = gather_feature_paths(data_path, subset)
    records = list(pool.imap(read_metadata_record, raw_feature_iterator(feature_paths)))
    return (
        pl.DataFrame(records)
        .with_columns(pl.lit(subset).alias("subset"))
        .select(ORDERED_COLUMNS)
    )


def read_metadata(data_dir: Path | str, output_path: Path | str | None = None) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """
    Write metadata to a parquet file and return train/test/challenge dataframes.
    """
    data_path: Path = Path(data_dir)
    parquet_path = Path(output_path) if output_path is not None else data_path / METADATA_FILENAME

    with multiprocessing.Pool() as pool:
        train_metadf = read_metadata_subset(data_path, "train", pool)
        test_metadf = read_metadata_subset(data_path, "test", pool)
        challenge_metadf = read_metadata_subset(data_path, "challenge", pool)

    pl.concat([train_metadf, test_metadf, challenge_metadf], how="vertical_relaxed").write_parquet(parquet_path)
    return train_metadf, test_metadf, challenge_metadf


def optimize_model(data_dir: Path | str) -> dict:
    """
    Run a grid search to find the best LightGBM parameters
    """
    # Read data
    X_train, y_train = read_vectorized_features(data_dir, "train")
    train_rows = y_train != -1
    X_train_labeled = X_train[train_rows]
    y_train_labeled = y_train[train_rows]

    # Score by ROC AUC
    # We're interested in low FPR rates, so we'll consider only the AUC for FPRs in [0,5e-3]
    score = make_scorer(roc_auc_score, max_fpr=5e-3)

    # Each row in X_train appears in chronological order of "first_seen_date" so this works for
    # progrssive time series splitting
    progressive_cv = TimeSeriesSplit(n_splits=3).split(X_train_labeled)

    fit_params = {"categorical_feature": [2, 3, 4, 5, 6, 701, 702]}
    param_grid = {
        "boosting_type": ["gbdt"],
        "objective": ["binary"],
        "num_iterations": [500, 1000],
        "learning_rate": [0.005, 0.05],
        "num_leaves": [512, 1024, 2048],
        "feature_fraction": [0.5, 0.8, 1.0],
        "bagging_fraction": [0.5, 0.8, 1.0],
    }
    grid = GridSearchCV(
        estimator=lgb.LGBMClassifier(n_jobs=-1, verbose=-1),
        cv=progressive_cv,
        param_grid=param_grid,
        scoring=score,
        n_jobs=1,
        verbose=3,
    )
    grid.fit(X_train_labeled, y_train_labeled, **fit_params)

    return grid.best_params_


def train_model(data_dir: Path | str, params: dict = {}) -> lgb.Booster:
    """
    Train LightGBM model on the vectorized features.
    """
    # Read data
    X, y = read_vectorized_features(data_dir, "train")
    feature_names = get_feature_names()

    # Verify that y_train is not formatted for multi-label classification
    if len(y.shape) != 1:
        raise ValueError("Encounted y_train with invalid shape. Use train_ovr_model() instead.")

    # Ignore files without a label/tag
    num_classes = np.max(y) + 1
    X = X[y != -1, :]
    y = y[y != -1]

    # Use a stratified split to make a validation set
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, stratify=y)
    train_set = lgb.Dataset(X_train, y_train, feature_name=feature_names, categorical_feature=[2, 3, 4, 5, 6, 701, 702])
    val_set = lgb.Dataset(X_val, y_val, reference=train_set, feature_name=feature_names, categorical_feature=[2, 3, 4, 5, 6, 701, 702])

    # Binary classification
    if num_classes == 2:
        return lgb.train(params, train_set, valid_sets=val_set)

    # Multiclass classification
    lgbm_params = {
        "objective": "multiclass",
        "num_class": num_classes,
        "metric": "multi_logloss"
    }
    params.update(lgbm_params)
    return lgb.train(params, train_set, valid_sets=val_set)


def train_ovr_model(data_dir: Path | str, params: dict = {}) -> lgb.Booster:
    """
    Returns a list of One-vs-Rest (OvR) LightGBM classifiers trained on the vectorized features.
    """
    # Read data
    X, y = read_vectorized_features(data_dir, "train")
    feature_names = get_feature_names()

    # Verify that y_train is not formatted for multi-label classification
    if len(y.shape) != 2:
        raise ValueError("Encounted y_train with invalid shape. Use train_model() instead.")

    # OvR Multilabel classification
    lgbm_models = []
    for i in range(y.shape[1]):
        lgbm_params = {
            "objective": "binary",
            "is_unbalance": True,
        }
        params.update(lgbm_params)
        y_i = y[:, i]
        X_train, X_val, y_train, y_val = train_test_split(X, y_i, test_size=0.1, stratify=y_i)
        train_set = lgb.Dataset(X_train, y_train, feature_name=feature_names, categorical_feature=[2, 3, 4, 5, 6, 701, 702])
        val_set = lgb.Dataset(X_val, y_val, reference=train_set, feature_name=feature_names, categorical_feature=[2, 3, 4, 5, 6, 701, 702])
        lgbm_models.append(lgb.train(params, train_set, valid_sets=val_set))
    return lgbm_models


def predict_sample(lgbm_model: lgb.Booster, file_data: bytes) -> float:
    """
    Predict a PE file with an LightGBM model
    """
    extractor = PEFeatureExtractor()
    features = np.array(extractor.feature_vector(file_data), dtype=np.float32)
    predict_result: np.ndarray = lgbm_model.predict([features])
    return float(predict_result[0])
