"""The frontend's half of the private conversation-offer channel.

A native conversation store can only reuse a history the frontend prepared: the `scope=` key on a generation request
says which conversation namespace a prompt belongs to, but by then the request is already admitted and running, so
nothing can be read from disk ahead of it.  A queued text request is therefore prepared (resolved, rendered and
tokenized) before it is admitted - only for the queue frontier the native side advertises - and *offered* to the
store over a private inherited socket, so the store can validate a matching history and read it while the request is
still waiting.  The request's own generation consumes exactly that prepared result (no re-render).

The channel is an inherited Linux `SOCK_SEQPACKET` socketpair, not a listening socket: the frontend creates the pair
before the engine is spawned, passes the child's end through `pass_fds` and names its number in
``STRATA_CONVERSATION_OFFER_FD``.  The engine answers with a HELLO carrying the execution domain's epoch, the queue
frontier it accepts and - the only place a persistent issuer key ever appears - the key that authorizes a stable
conversation scope.  Key bytes are never logged, never written to disk and never sent to a client.

Fixed 72-byte little-endian header (the native side's `conversation_offers.hpp`)::

    struct.pack("<4sHHQQQII32s", magic, version, type, request_id, generation, deadline_monotonic_ns,
                token_count, flags, scope)

* HELLO (type 1, child to frontend): `request_id=0`, `generation` is the domain epoch, the deadline field is the
  metadata budget in bytes, `token_count` is the queue frontier (1 or 2), flags bit 0 says the store persists
  histories and `scope` is the 32-byte issuer key (all zero when it does not).
* OFFER (type 2, frontend to child): nonzero request id and generation, the deadline field is the original request's
  deadline in monotonic nanoseconds, `token_count` is the payload's token count, flags bit 0 asks for the request's
  control vector, `scope` is the authorized 32-byte conversation scope, and one `SCM_RIGHTS` descriptor carries
  exactly `token_count` little-endian int32 tokens, write-sealed and read-only.
* CANCEL (type 3, frontend to child): the same request id and generation, no descriptor.

Everything here is advisory.  An offer that cannot be sent, a store that is not enabled, a channel that the engine
never answered on: each is an ordinary cache miss, and the request is prepared and run at demand exactly as it would
have been.  Nothing in this module may block a request: the send is bounded and the reader only ever sets state.
"""
from __future__ import annotations

import array
import dataclasses
import fcntl
import hashlib
import os
import socket
import struct
import sys
import threading
from typing import Sequence

MAGIC = b"SCC1"
VERSION = 1
TYPE_HELLO = 1
TYPE_OFFER = 2
TYPE_CANCEL = 3

HEADER = struct.Struct("<4sHHQQQII32s")
HEADER_BYTES = HEADER.size              # 72
SCOPE_BYTES = 32

FLAG_STORE_ENABLED = 1 << 0             # HELLO: the store persists histories (a zero key means it does not)
FLAG_WANT_CVEC = 1 << 0                 # OFFER: this request wants its control vector priced

OFFER_FD_ENV = "STRATA_CONVERSATION_OFFER_FD"
SEND_TIMEOUT_S = 0.5                    # an advisory send is bounded: it may never hold a queued request for long
CLOSE_TIMEOUT_S = 2.0                   # how long close() waits for the reader thread

# The channel needs a Linux socketpair of SEQPACKET sockets and memfd payloads.  Anywhere else the frontend keeps
# its ordinary demand path: no offers, no persistent scope, no fd inherited by the engine.
SUPPORTED = (os.name == "posix" and sys.platform.startswith("linux") and hasattr(os, "memfd_create")
             and hasattr(socket, "SOCK_SEQPACKET"))


class OfferProtocolError(ValueError):
    """A malformed message on the private channel.  Never a client's fault, so never a client's error."""


@dataclasses.dataclass(frozen=True)
class Hello:
    """What the engine's startup message says about this incarnation."""

    generation: int                     # the execution domain's epoch (nonzero)
    frontier: int                       # how many queued requests may be prepared ahead of admission (1 or 2)
    metadata_bytes: int                 # the byte budget for outstanding token descriptors
    enabled: bool                       # flags bit 0: the store persists histories
    issuer_key: bytes                   # 32 bytes; all zero when persistence is disabled

    @property
    def store_enabled(self) -> bool:
        """True only when a persistent store vouches for the key: a ZeroKey HELLO means RAM-only serving."""
        return bool(self.enabled) and any(self.issuer_key)


def parse_hello(payload: bytes) -> Hello:
    """The engine's HELLO, strictly validated (a malformed one is refused, never half-used)."""
    if len(payload) != HEADER_BYTES:
        raise OfferProtocolError(f"a conversation-offer message is exactly {HEADER_BYTES} bytes, not {len(payload)}")
    magic, version, kind, request_id, generation, deadline_ns, token_count, flags, scope = HEADER.unpack(payload)
    if magic != MAGIC:
        raise OfferProtocolError("not a conversation-offer message")
    if version != VERSION:
        raise OfferProtocolError(f"conversation-offer protocol version {version} is not supported")
    if kind != TYPE_HELLO:
        raise OfferProtocolError(f"expected a HELLO, got type {kind}")
    if request_id != 0:
        raise OfferProtocolError("a HELLO carries no request id")
    if flags & ~FLAG_STORE_ENABLED:
        raise OfferProtocolError("unknown HELLO flags")
    if generation == 0:
        raise OfferProtocolError("a HELLO must carry its domain epoch")
    if token_count not in (1, 2):
        raise OfferProtocolError("the prepared-queue frontier is one request, or two")
    return Hello(generation=generation, frontier=token_count, metadata_bytes=deadline_ns,
                 enabled=bool(flags & FLAG_STORE_ENABLED), issuer_key=scope)


def hello_payload(hello: Hello) -> bytes:
    """The wire form of a HELLO.  Used by this module's tests and by anything speaking the native side."""
    flags = FLAG_STORE_ENABLED if hello.enabled else 0
    return HEADER.pack(MAGIC, VERSION, TYPE_HELLO, 0, hello.generation, hello.metadata_bytes, hello.frontier, flags,
                       hello.issuer_key)


def sealed_token_payload(tokens: Sequence[int]) -> int:
    """A sealed, read-only memfd holding exactly `tokens` as little-endian int32: the caller closes the descriptor.

    Sealing (SHRINK/GROW/WRITE/SEAL) fixes the length and the bytes: the store may map it and read it, and no one -
    the frontend included - can change it afterwards.  The descriptor handed over is a read-only duplicate, so the
    receiver cannot be given a writable payload by accident (`/proc` is not always mounted: then the file's own
    write seal is the guarantee, and the original descriptor is returned).
    """
    data = array.array("i", [int(t) for t in tokens])
    if data.itemsize != 4:
        raise OfferProtocolError("this interpreter's C int is not 32 bits")
    if sys.byteorder != "little":
        data.byteswap()
    fd = os.memfd_create("strata-conversation-offer", os.MFD_CLOEXEC | os.MFD_ALLOW_SEALING)
    try:
        view = memoryview(data.tobytes())
        while view:
            written = os.write(fd, view)
            if written <= 0:
                raise OSError("the token payload could not be written")
            view = view[written:]
        os.lseek(fd, 0, os.SEEK_SET)
        fcntl.fcntl(fd, fcntl.F_ADD_SEALS,
                    fcntl.F_SEAL_SHRINK | fcntl.F_SEAL_GROW | fcntl.F_SEAL_WRITE | fcntl.F_SEAL_SEAL)
        read_only = _read_only_dup(fd)
    except BaseException:
        os.close(fd)
        raise
    if read_only is None:
        return fd
    os.close(fd)
    return read_only


def _read_only_dup(fd: int) -> int | None:
    """A read-only descriptor for the same sealed payload, or None when this system cannot reopen it."""
    try:
        return os.open(f"/proc/self/fd/{fd}", os.O_RDONLY | os.O_CLOEXEC)
    except OSError:
        return None


def generation_digest(messages, tools, kwargs, max_new, force) -> str:
    """A digest of everything that decides a request's prepared ids and tokens.

    A prepared result is reused only when the generation asks with exactly these inputs, so a re-resolved request
    (a later turn, a second tool round of the same request) always renders its own prompt.
    """
    try:
        blob = repr((messages, tools, kwargs, max_new, force))
    except Exception:                    # noqa: BLE001 - a value that cannot be printed is never reused
        return ""
    return hashlib.sha256(blob.encode("utf-8", "surrogatepass")).hexdigest()


@dataclasses.dataclass
class PreparedGeneration:
    """One request's prepared prompt, frozen before admission and consumed once by the generation that asked for it.

    `offer_id` is nonzero only when the payload also reached the native store; `withdrawn` and `consumed` keep the
    offer's lifetime tied to this request (its deadline, its client and its cancellation) instead of the request
    object's.
    """

    ids: tuple[int, ...]
    thinking: bool
    max_new: int
    digest: str
    ctx: int                                   # the engine's context at the time it was prepared
    link: "ConversationOfferLink | None" = None
    offer_id: int = 0
    generation: int = 0
    consumed: bool = False
    withdrawn: bool = False

    @property
    def offered(self) -> bool:
        return self.offer_id > 0

    def reusable(self, digest: str, ctx: int) -> bool:
        """True when the generation asking for this result asks for exactly it, in the same engine incarnation."""
        return not self.consumed and self.digest == digest and self.ctx == ctx


class ConversationOfferLink:
    """The frontend's end of one engine incarnation's private channel.

    Create it before the engine is spawned (the child's end is what the engine inherits), `start()` its reader once
    the process exists, then `offer()`/`cancel()`.  `close()` releases both ends and is idempotent.  Memory bounds
    are the frontier (only that many requests are ever offered, and the native side refuses the rest) and the byte
    budget the HELLO states; every refusal is counted in `stats()`.
    """

    def __init__(self, send_timeout_s: float = SEND_TIMEOUT_S):
        parent, child = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
        parent.settimeout(send_timeout_s)
        self._lock = threading.Lock()
        self._parent = parent
        self._child: socket.socket | None = child
        self._child_fd = child.fileno()
        self._hello: Hello | None = None
        self._hello_event = threading.Event()
        self._closed = False
        self._reader: threading.Thread | None = None
        self._outstanding: dict[int, tuple[int, int]] = {}          # request id -> (generation, payload bytes)
        self._outstanding_bytes = 0
        self._next_id = 0
        self.accepted = 0                       # offers the store was told about
        self.rejected = 0                       # offers this side refused to send
        self.cancelled = 0                      # cancellations sent
        self.accepted_token_bytes = 0
        self.received_fds = 0                   # descriptors the engine sent us (never expected: closed at once)
        self.malformed = 0                      # messages that failed validation
        self._reader_error: str | None = None

    # -- the child's end: what the engine inherits ------------------------------------------------------------------

    @property
    def child_fd(self) -> int:
        """The number the engine is told through `STRATA_CONVERSATION_OFFER_FD` (and inherits through pass_fds)."""
        return self._child_fd

    @property
    def child_socket(self) -> socket.socket | None:
        """The child's socket object, for an embedder (or a test) that speaks the native side itself."""
        return self._child

    def pass_fds(self) -> tuple[int, ...]:
        return (self._child_fd,)

    def child_environment(self, env: dict | None = None) -> dict:
        """The engine's environment with this incarnation's descriptor number added (a copy, never the caller's)."""
        out = dict(os.environ if env is None else env)
        out[OFFER_FD_ENV] = str(self._child_fd)
        return out

    def release_child(self) -> None:
        """Drop this process's copy of the child's end once the engine owns it (idempotent, never raises)."""
        with self._lock:
            child, self._child = self._child, None
        if child is not None:
            try:
                child.close()
            except OSError:
                pass

    # -- state the engine published ---------------------------------------------------------------------------------

    def start(self, name: str = "strata-conversation-offers") -> None:
        """Begin reading the engine's messages (one HELLO per incarnation, then only closes)."""
        with self._lock:
            if self._reader is not None or self._closed:
                return
            self._reader = threading.Thread(target=self._read_loop, name=name, daemon=True)
        self._reader.start()

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def hello(self) -> Hello | None:
        return self._hello

    def wait_hello(self, timeout: float) -> bool:
        """True once the engine's HELLO has been read (`timeout` may be None to wait for it indefinitely)."""
        return self._hello_event.wait(timeout)

    @property
    def store_enabled(self) -> bool:
        hello = self._hello
        return hello is not None and hello.store_enabled

    def scope_key(self) -> bytes | None:
        """The issuer key that makes conversation scopes stable across restarts, or None (RAM-only serving)."""
        hello = self._hello
        return hello.issuer_key if hello is not None and hello.store_enabled else None

    def epoch(self) -> int:
        return self._hello.generation if self._hello is not None else 0

    def frontier(self) -> int:
        return self._hello.frontier if self._hello is not None else 0

    def metadata_bytes(self) -> int:
        return self._hello.metadata_bytes if self._hello is not None else 0

    # -- what the frontend sends ------------------------------------------------------------------------------------

    def next_request_id(self) -> int:
        """A nonzero id no offer of this incarnation has used (the native side refuses a duplicate)."""
        with self._lock:
            self._next_id += 1
            return self._next_id

    def offer(self, request_id: int, generation: int, deadline_ns: int, scope: str, tokens: Sequence[int],
              token_limit: int = 0, want_cvec: bool = True) -> bool:
        """Offer `tokens` as this request's prepared prompt.  -> whether the store was told about them.

        Refused (and counted) when the store is not enabled, the ids do not belong to this incarnation, the payload
        is empty or over `token_limit`, the byte budget is full, the descriptor cannot be built, or the send fails.
        A refusal is never an error: the request is simply prepared and run at demand.
        """
        try:
            scope_bytes = _scope_bytes(scope)
        except OfferProtocolError:
            with self._lock:
                self.rejected += 1
            return False
        if request_id <= 0 or generation <= 0 or generation != self.epoch() or not self.store_enabled:
            with self._lock:
                self.rejected += 1
            return False
        count = len(tokens)
        if count < 1 or (token_limit and count > token_limit):
            with self._lock:
                self.rejected += 1
            return False
        nbytes = 4 * count
        with self._lock:
            if self._closed or request_id in self._outstanding:
                self.rejected += 1
                return False
            budget = self.metadata_bytes()
            if budget and self._outstanding_bytes + nbytes > budget:
                self.rejected += 1
                return False
            self._outstanding[request_id] = (generation, nbytes)
            self._outstanding_bytes += nbytes
        flags = FLAG_WANT_CVEC if want_cvec else 0
        try:
            fd = sealed_token_payload(tokens)
        except (OSError, OfferProtocolError, ValueError, OverflowError) as e:
            self._release(request_id)
            with self._lock:
                self.rejected += 1
            self._note(f"the prepared payload could not be sealed: {e}")
            return False
        try:
            payload = HEADER.pack(MAGIC, VERSION, TYPE_OFFER, request_id, generation, int(deadline_ns), count, flags,
                                  scope_bytes)
            if not self._send(payload, fd):
                self._release(request_id)
                with self._lock:
                    self.rejected += 1
                return False
        except struct.error:
            self._release(request_id)
            with self._lock:
                self.rejected += 1
            return False
        finally:
            try:
                os.close(fd)
            except OSError:
                pass
        with self._lock:
            self.accepted += 1
            self.accepted_token_bytes += nbytes
        return True

    def cancel(self, request_id: int, generation: int) -> bool:
        """Retire exactly the offered request/generation.  -> whether the store was told.

        Only a pair this link really offered is cancelled, and only with the generation it was offered under: an
        unknown, already retired or mismatched pair is left alone (and stays outstanding, as the store knows it).
        """
        with self._lock:
            if self._closed:
                return False
            entry = self._outstanding.get(request_id)
            if entry is None or entry[0] != generation:
                return False
            del self._outstanding[request_id]
            self._outstanding_bytes = max(0, self._outstanding_bytes - entry[1])
        payload = HEADER.pack(MAGIC, VERSION, TYPE_CANCEL, request_id, generation, 0, 0, 0, b"\0" * SCOPE_BYTES)
        if not self._send(payload, None):
            return False
        with self._lock:
            self.cancelled += 1
        return True

    def stats(self) -> dict:
        with self._lock:
            hello = self._hello
            return {"store": hello is not None and hello.store_enabled, "frontier": hello.frontier if hello else 0,
                    "accepted": self.accepted, "cancelled": self.cancelled, "rejected": self.rejected,
                    "accepted_token_bytes": self.accepted_token_bytes, "outstanding": len(self._outstanding),
                    "outstanding_bytes": self._outstanding_bytes, "malformed": self.malformed,
                    "received_fds": self.received_fds, "closed": self._closed, "reader_error": self._reader_error}

    def close(self) -> None:
        """Close both ends and stop the reader.  Idempotent; the engine's process is ended by its owner, not here."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._outstanding.clear()
            self._outstanding_bytes = 0
            reader, self._reader = self._reader, None
        self.release_child()
        try:
            self._parent.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        try:
            self._parent.close()
        except OSError:
            pass
        self._hello_event.set()
        if reader is not None and reader is not threading.current_thread():
            reader.join(timeout=CLOSE_TIMEOUT_S)

    # -- internals -------------------------------------------------------------------------------------------------

    def _note(self, message: str) -> None:
        """One bounded diagnostic line per failure kind; never key or prompt bytes."""
        print(f"[strata] conversation offers: {message}", flush=True)

    def _release(self, request_id: int) -> None:
        """Forget a reservation this side could not send; the store was never told about it."""
        with self._lock:
            entry = self._outstanding.pop(request_id, None)
            if entry is not None:
                self._outstanding_bytes = max(0, self._outstanding_bytes - entry[1])

    def _send(self, payload: bytes, fd: int | None) -> bool:
        """One bounded message.  False when this incarnation's channel is gone (never an exception to the caller)."""
        if self._closed:
            return False
        try:
            if fd is None:
                self._parent.send(payload)
            else:
                fds = array.array("i", [fd])
                self._parent.sendmsg([payload], [(socket.SOL_SOCKET, socket.SCM_RIGHTS, fds)])
        except (OSError, socket.timeout):
            return False
        return True

    def _read_loop(self) -> None:
        """Read the engine's messages: the HELLO, then only a close.  Never raises out of the thread."""
        while True:
            with self._lock:
                if self._closed:
                    return
            try:
                payload, fds = self._recv()
            except OSError as e:
                with self._lock:
                    if not self._closed:                     # a close is not an error worth remembering
                        self._reader_error = str(e)
                return
            if payload is None:
                return                                   # the engine closed its end (or this link was closed)
            for fd in fds:                               # the store never sends descriptors to us: close them at once
                try:
                    os.close(fd)
                except OSError:
                    pass
                with self._lock:
                    self.received_fds += 1
            if fds or not payload:                       # only a read timeout and an unexpected descriptor land here
                with self._lock:
                    self.malformed += 1 if fds else 0
                continue
            if self._hello is not None:
                with self._lock:
                    self.malformed += 1                  # one HELLO per incarnation
                continue
            try:
                hello = parse_hello(payload)
            except OfferProtocolError as e:
                with self._lock:
                    self.malformed += 1
                self._note(f"a message from the engine was refused: {e}")
                continue
            with self._lock:
                self._hello = hello
            self._hello_event.set()
            self._note(("the engine's conversation store is enabled (frontier %d, %d metadata bytes, epoch %d)"
                        % (hello.frontier, hello.metadata_bytes, hello.generation)) if hello.store_enabled else
                       "the engine has no persistent conversation store: prompts stay in this process")

    def _recv(self) -> tuple[bytes | None, list[int]]:
        """One datagram and any descriptors it carried (None when the peer closed its end)."""
        try:
            payload, ancillary, _flags, _address = self._parent.recvmsg(HEADER_BYTES, socket.CMSG_SPACE(4 * 8))
        except socket.timeout:
            return b"", []
        except OSError:
            raise
        if not payload:
            return None, []
        fds: list[int] = []
        for level, kind, data in ancillary:
            if level == socket.SOL_SOCKET and kind == socket.SCM_RIGHTS:
                fds.extend(array.array("i", data[:len(data) - (len(data) % 4)]).tolist())
        return payload, fds


def _scope_bytes(scope: str) -> bytes:
    """The 32-byte scope field of an authorized conversation scope (64 lowercase hex, never all zero)."""
    if not isinstance(scope, str) or len(scope) != 2 * SCOPE_BYTES:
        raise OfferProtocolError("a conversation scope is 64 hex digits")
    try:
        raw = bytes.fromhex(scope)
    except ValueError:
        raise OfferProtocolError("a conversation scope is 64 hex digits") from None
    if not any(raw):
        raise OfferProtocolError("a conversation scope is never all zero")
    return raw
