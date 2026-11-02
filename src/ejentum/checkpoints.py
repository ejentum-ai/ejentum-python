"""
Ejentum Distributed Checkpoint Store.
Supports Memory, Local File, Redis, Amazon S3, and Google Cloud Storage (GCS) backends.
Provides state snapshotting and automated time-travel rollbacks on invariant failures.
"""

from __future__ import annotations

import abc
import json
import os
import threading
import time
from typing import Any, Dict, List, Optional, Union


class BaseCheckpointStore(abc.ABC):
    """Abstract base class for Ejentum state checkpoint stores."""

    @abc.abstractmethod
    def save(
        self,
        thread_id: str,
        state: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
        checkpoint_id: Optional[str] = None,
    ) -> str:
        """Save an execution state checkpoint and return its checkpoint_id."""
        raise NotImplementedError

    @abc.abstractmethod
    def load(self, checkpoint_id: str) -> Optional[Dict[str, Any]]:
        """Load a checkpoint record by checkpoint_id."""
        raise NotImplementedError

    @abc.abstractmethod
    def list_checkpoints(self, thread_id: str, limit: int = 20) -> List[Dict[str, Any]]:
        """List checkpoints for a thread ordered newest first."""
        raise NotImplementedError

    @abc.abstractmethod
    def get_latest(self, thread_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve the most recent checkpoint for a thread."""
        raise NotImplementedError

    def rollback(self, thread_id: str, steps: int = 1) -> Optional[Dict[str, Any]]:
        """Roll back execution state to N steps prior.

        Args:
            thread_id: Workflow or agent thread identifier.
            steps: Number of checkpoints to step backwards (default: 1, i.e., node N-1).

        Returns:
            The prior checkpoint record or None if insufficient history exists.
        """
        history = self.list_checkpoints(thread_id, limit=steps + 1)
        if len(history) <= steps:
            return history[-1] if history else None
        target_cp = history[steps]
        return self.load(target_cp["checkpoint_id"])

    @abc.abstractmethod
    def delete(self, checkpoint_id: str) -> bool:
        """Delete a checkpoint by ID."""
        raise NotImplementedError


class MemoryCheckpointStore(BaseCheckpointStore):
    """Thread-safe In-Memory Checkpoint Store.

    Ideal for local execution, unit testing, and fast single-process agent loops.
    """

    def __init__(self, max_history_per_thread: int = 100):
        self.max_history = max(1, int(max_history_per_thread))
        self._checkpoints: Dict[str, Dict[str, Any]] = {}
        self._threads: Dict[str, List[str]] = {}
        self._lock = threading.Lock()

    def save(
        self,
        thread_id: str,
        state: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
        checkpoint_id: Optional[str] = None,
    ) -> str:
        ts = time.time()
        cp_id = checkpoint_id or f"cp_{int(ts * 1000)}_{len(self._checkpoints) + 1}"
        record = {
            "checkpoint_id": cp_id,
            "thread_id": thread_id,
            "state": dict(state),
            "metadata": dict(metadata or {}),
            "timestamp": ts,
        }
        with self._lock:
            self._checkpoints[cp_id] = record
            if thread_id not in self._threads:
                self._threads[thread_id] = []
            self._threads[thread_id].append(cp_id)
            if len(self._threads[thread_id]) > self.max_history:
                evicted_id = self._threads[thread_id].pop(0)
                self._checkpoints.pop(evicted_id, None)
        return cp_id

    def load(self, checkpoint_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            record = self._checkpoints.get(checkpoint_id)
            return dict(record) if record else None

    def list_checkpoints(self, thread_id: str, limit: int = 20) -> List[Dict[str, Any]]:
        with self._lock:
            cp_ids = list(self._threads.get(thread_id, []))
        res = []
        for cp_id in reversed(cp_ids[-limit:]):
            rec = self.load(cp_id)
            if rec:
                res.append(rec)
        return res

    def get_latest(self, thread_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            cp_ids = self._threads.get(thread_id, [])
            if not cp_ids:
                return None
            latest_id = cp_ids[-1]
        return self.load(latest_id)

    def delete(self, checkpoint_id: str) -> bool:
        with self._lock:
            if checkpoint_id in self._checkpoints:
                rec = self._checkpoints.pop(checkpoint_id)
                t_id = rec.get("thread_id")
                if t_id and t_id in self._threads and checkpoint_id in self._threads[t_id]:
                    self._threads[t_id].remove(checkpoint_id)
                return True
            return False

    def clear(self) -> None:
        with self._lock:
            self._checkpoints.clear()
            self._threads.clear()


class FileCheckpointStore(BaseCheckpointStore):
    """Local Filesystem Checkpoint Store.

    Persists state snapshots as JSON documents on persistent disk or container volumes.
    """

    def __init__(self, dir_path: Optional[str] = None):
        base_dir = dir_path or os.getenv("EJENTUM_CHECKPOINT_DIR", "data/checkpoints")
        self.dir_path = os.path.abspath(base_dir)
        os.makedirs(self.dir_path, exist_ok=True)
        self._lock = threading.Lock()

    def _thread_dir(self, thread_id: str) -> str:
        safe_id = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in thread_id)
        tdir = os.path.join(self.dir_path, safe_id)
        os.makedirs(tdir, exist_ok=True)
        return tdir

    def save(
        self,
        thread_id: str,
        state: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
        checkpoint_id: Optional[str] = None,
    ) -> str:
        ts = time.time()
        cp_id = checkpoint_id or f"cp_{int(ts * 1000)}"
        tdir = self._thread_dir(thread_id)
        file_path = os.path.join(tdir, f"{cp_id}.json")
        record = {
            "checkpoint_id": cp_id,
            "thread_id": thread_id,
            "state": state,
            "metadata": metadata or {},
            "timestamp": ts,
        }
        with self._lock:
            tmp_path = f"{file_path}.tmp_{os.getpid()}_{time.perf_counter_ns()}"
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(record, f, indent=2, default=str)
            os.replace(tmp_path, file_path)
        return cp_id

    def load(self, checkpoint_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            for root, _, files in os.walk(self.dir_path):
                target = f"{checkpoint_id}.json"
                if target in files:
                    full_p = os.path.join(root, target)
                    try:
                        with open(full_p, "r", encoding="utf-8") as f:
                            return json.load(f)
                    except Exception:
                        return None
        return None

    def list_checkpoints(self, thread_id: str, limit: int = 20) -> List[Dict[str, Any]]:
        tdir = self._thread_dir(thread_id)
        with self._lock:
            if not os.path.exists(tdir):
                return []
            files = [f for f in os.listdir(tdir) if f.endswith(".json")]
            files.sort(reverse=True)
            res = []
            for fname in files[:limit]:
                fpath = os.path.join(tdir, fname)
                try:
                    with open(fpath, "r", encoding="utf-8") as f:
                        res.append(json.load(f))
                except Exception:
                    continue
            return res

    def get_latest(self, thread_id: str) -> Optional[Dict[str, Any]]:
        results = self.list_checkpoints(thread_id, limit=1)
        return results[0] if results else None

    def delete(self, checkpoint_id: str) -> bool:
        with self._lock:
            for root, _, files in os.walk(self.dir_path):
                target = f"{checkpoint_id}.json"
                if target in files:
                    try:
                        os.remove(os.path.join(root, target))
                        return True
                    except OSError:
                        return False
        return False


class RedisCheckpointStore(BaseCheckpointStore):
    """Distributed Redis Checkpoint Store for multi-instance agent workers.

    Coordinates checkpoints across stateless containers or distributed worker nodes.
    Supports redis.Redis client or direct connection URL.
    """

    def __init__(
        self,
        redis_url: Optional[str] = None,
        redis_client: Optional[Any] = None,
        prefix: str = "ejentum:cp",
        ttl_seconds: int = 604800,
    ):
        self.prefix = prefix.rstrip(":")
        self.ttl = ttl_seconds
        if redis_client is not None:
            self._client = redis_client
        else:
            url = redis_url or os.getenv("EJENTUM_REDIS_URL", "redis://localhost:6379/0")
            try:
                import redis
                self._client = redis.Redis.from_url(url, decode_responses=True)
            except ImportError:
                raise ImportError(
                    "The 'redis' package is required for RedisCheckpointStore. "
                    "Install it via `pip install redis`."
                )

    def _key(self, checkpoint_id: str) -> str:
        return f"{self.prefix}:records:{checkpoint_id}"

    def _thread_set_key(self, thread_id: str) -> str:
        return f"{self.prefix}:threads:{thread_id}"

    def save(
        self,
        thread_id: str,
        state: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
        checkpoint_id: Optional[str] = None,
    ) -> str:
        ts = time.time()
        cp_id = checkpoint_id or f"cp_{int(ts * 1000)}"
        record = {
            "checkpoint_id": cp_id,
            "thread_id": thread_id,
            "state": state,
            "metadata": metadata or {},
            "timestamp": ts,
        }
        raw_val = json.dumps(record, default=str)
        key = self._key(cp_id)
        thread_key = self._thread_set_key(thread_id)

        self._client.set(key, raw_val, ex=self.ttl)
        self._client.zadd(thread_key, {cp_id: ts})
        self._client.expire(thread_key, self.ttl)
        return cp_id

    def load(self, checkpoint_id: str) -> Optional[Dict[str, Any]]:
        val = self._client.get(self._key(checkpoint_id))
        if not val:
            return None
        try:
            return json.loads(val)
        except Exception:
            return None

    def list_checkpoints(self, thread_id: str, limit: int = 20) -> List[Dict[str, Any]]:
        thread_key = self._thread_set_key(thread_id)
        cp_ids = self._client.zrevrange(thread_key, 0, max(0, limit - 1))
        results = []
        for cp_id in cp_ids:
            rec = self.load(cp_id)
            if rec:
                results.append(rec)
        return results

    def get_latest(self, thread_id: str) -> Optional[Dict[str, Any]]:
        thread_key = self._thread_set_key(thread_id)
        latest_ids = self._client.zrevrange(thread_key, 0, 0)
        if not latest_ids:
            return None
        return self.load(latest_ids[0])

    def delete(self, checkpoint_id: str) -> bool:
        rec = self.load(checkpoint_id)
        if rec and "thread_id" in rec:
            self._client.zrem(self._thread_set_key(rec["thread_id"]), checkpoint_id)
        return bool(self._client.delete(self._key(checkpoint_id)))


class S3CheckpointStore(BaseCheckpointStore):
    """Amazon S3 Durable Checkpoint Store.

    Stores execution checkpoints in Amazon S3 for compliance auditing and disaster recovery.
    """

    def __init__(
        self,
        bucket_name: Optional[str] = None,
        prefix: str = "checkpoints",
        s3_client: Optional[Any] = None,
    ):
        self.bucket = bucket_name or os.getenv("EJENTUM_S3_BUCKET", "")
        if not self.bucket:
            raise ValueError(
                "bucket_name or EJENTUM_S3_BUCKET environment variable must be specified for S3CheckpointStore."
            )
        self.prefix = prefix.strip("/")
        if s3_client is not None:
            self._client = s3_client
        else:
            try:
                import boto3
                self._client = boto3.client("s3")
            except ImportError:
                raise ImportError(
                    "The 'boto3' package is required for S3CheckpointStore. "
                    "Install it via `pip install boto3`."
                )

    def _s3_key(self, thread_id: str, checkpoint_id: str) -> str:
        return f"{self.prefix}/{thread_id}/{checkpoint_id}.json"

    def save(
        self,
        thread_id: str,
        state: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
        checkpoint_id: Optional[str] = None,
    ) -> str:
        ts = time.time()
        cp_id = checkpoint_id or f"cp_{int(ts * 1000)}"
        key = self._s3_key(thread_id, cp_id)
        record = {
            "checkpoint_id": cp_id,
            "thread_id": thread_id,
            "state": state,
            "metadata": metadata or {},
            "timestamp": ts,
        }
        body = json.dumps(record, indent=2, default=str).encode("utf-8")
        self._client.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=body,
            ContentType="application/json"
        )
        return cp_id

    def load(self, checkpoint_id: str) -> Optional[Dict[str, Any]]:
        prefix = f"{self.prefix}/"
        try:
            resp = self._client.list_objects_v2(Bucket=self.bucket, Prefix=prefix)
            for item in resp.get("Contents", []):
                if item["Key"].endswith(f"/{checkpoint_id}.json"):
                    obj = self._client.get_object(Bucket=self.bucket, Key=item["Key"])
                    return json.loads(obj["Body"].read().decode("utf-8"))
        except Exception:
            return None
        return None

    def list_checkpoints(self, thread_id: str, limit: int = 20) -> List[Dict[str, Any]]:
        prefix = f"{self.prefix}/{thread_id}/"
        try:
            resp = self._client.list_objects_v2(Bucket=self.bucket, Prefix=prefix)
            items = sorted(resp.get("Contents", []), key=lambda x: x["LastModified"], reverse=True)
            results = []
            for item in items[:limit]:
                obj = self._client.get_object(Bucket=self.bucket, Key=item["Key"])
                results.append(json.loads(obj["Body"].read().decode("utf-8")))
            return results
        except Exception:
            return []

    def get_latest(self, thread_id: str) -> Optional[Dict[str, Any]]:
        res = self.list_checkpoints(thread_id, limit=1)
        return res[0] if res else None

    def delete(self, checkpoint_id: str) -> bool:
        prefix = f"{self.prefix}/"
        try:
            resp = self._client.list_objects_v2(Bucket=self.bucket, Prefix=prefix)
            for item in resp.get("Contents", []):
                if item["Key"].endswith(f"/{checkpoint_id}.json"):
                    self._client.delete_object(Bucket=self.bucket, Key=item["Key"])
                    return True
        except Exception:
            return False
        return False


class GCSCheckpointStore(BaseCheckpointStore):
    """Google Cloud Storage (GCS) Durable Checkpoint Store."""

    def __init__(
        self,
        bucket_name: Optional[str] = None,
        prefix: str = "checkpoints",
        gcs_client: Optional[Any] = None,
    ):
        self.bucket_name = bucket_name or os.getenv("EJENTUM_GCS_BUCKET", "")
        if not self.bucket_name:
            raise ValueError(
                "bucket_name or EJENTUM_GCS_BUCKET environment variable must be specified for GCSCheckpointStore."
            )
        self.prefix = prefix.strip("/")
        if gcs_client is not None:
            self._client = gcs_client
        else:
            try:
                from google.cloud import storage
                self._client = storage.Client()
            except ImportError:
                raise ImportError(
                    "The 'google-cloud-storage' package is required for GCSCheckpointStore. "
                    "Install it via `pip install google-cloud-storage`."
                )
        self._bucket = self._client.bucket(self.bucket_name)

    def _blob_name(self, thread_id: str, checkpoint_id: str) -> str:
        return f"{self.prefix}/{thread_id}/{checkpoint_id}.json"

    def save(
        self,
        thread_id: str,
        state: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
        checkpoint_id: Optional[str] = None,
    ) -> str:
        ts = time.time()
        cp_id = checkpoint_id or f"cp_{int(ts * 1000)}"
        blob = self._bucket.blob(self._blob_name(thread_id, cp_id))
        record = {
            "checkpoint_id": cp_id,
            "thread_id": thread_id,
            "state": state,
            "metadata": metadata or {},
            "timestamp": ts,
        }
        blob.upload_from_string(json.dumps(record, indent=2, default=str), content_type="application/json")
        return cp_id

    def load(self, checkpoint_id: str) -> Optional[Dict[str, Any]]:
        prefix = f"{self.prefix}/"
        try:
            blobs = self._client.list_blobs(self._bucket, prefix=prefix)
            for b in blobs:
                if b.name.endswith(f"/{checkpoint_id}.json"):
                    content = b.download_as_text()
                    return json.loads(content)
        except Exception:
            return None
        return None

    def list_checkpoints(self, thread_id: str, limit: int = 20) -> List[Dict[str, Any]]:
        prefix = f"{self.prefix}/{thread_id}/"
        try:
            blobs = list(self._client.list_blobs(self._bucket, prefix=prefix))
            blobs.sort(key=lambda x: x.updated or x.time_created, reverse=True)
            results = []
            for b in blobs[:limit]:
                results.append(json.loads(b.download_as_text()))
            return results
        except Exception:
            return []

    def get_latest(self, thread_id: str) -> Optional[Dict[str, Any]]:
        res = self.list_checkpoints(thread_id, limit=1)
        return res[0] if res else None

    def delete(self, checkpoint_id: str) -> bool:
        prefix = f"{self.prefix}/"
        try:
            blobs = self._client.list_blobs(self._bucket, prefix=prefix)
            for b in blobs:
                if b.name.endswith(f"/{checkpoint_id}.json"):
                    b.delete()
                    return True
        except Exception:
            return False
        return False


def create_checkpoint_store(
    backend: Optional[str] = None,
    **kwargs: Any
) -> BaseCheckpointStore:
    """Create a configured checkpoint store based on backend type or environment variable.

    Args:
        backend: 'memory', 'file', 'redis', 's3', or 'gcs'.
        **kwargs: Backend-specific arguments (dir_path, redis_url, bucket_name, prefix, etc.).

    Returns:
        Instance of BaseCheckpointStore.
    """
    selected = (backend or os.getenv("EJENTUM_CHECKPOINT_BACKEND", "memory")).lower().strip()
    if selected == "memory":
        return MemoryCheckpointStore(**kwargs)
    elif selected in ("file", "disk", "local"):
        return FileCheckpointStore(**kwargs)
    elif selected == "redis":
        return RedisCheckpointStore(**kwargs)
    elif selected in ("s3", "aws"):
        return S3CheckpointStore(**kwargs)
    elif selected in ("gcs", "google"):
        return GCSCheckpointStore(**kwargs)
    else:
        raise ValueError(
            f"Unknown checkpoint backend '{selected}'. Supported backends: 'memory', 'file', 'redis', 's3', 'gcs'."
        )
