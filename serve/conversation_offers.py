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

* HELLO (type 1, child to frontend): `request_id=0`, `generation` is the domain epoch (nonzero when the store is
  enabled; it may be zero without a persistent domain), the deadline field is the metadata budget in bytes,
  `token_count` is the queue frontier (1 or 2), flags bit 0 says the store persists histories, and `scope` is the
  32-byte issuer key (all zero when it does not).
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
import ctypes
import errno
import dataclasses
import hashlib
import json
import os
import socket
import struct
import sys
import threading
from typing import Sequence

try:
    import fcntl
except ImportError:                       # the ordinary Windows frontend has no private Linux channel
    fcntl = None

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
OFFER_ID_MAX = (1 << 63) - 1            # the private id space: native-only demand reads use the high bit, so an
                                        # offer is never at or above it (the two must not collide on any one key)
RECV_FLAGS = getattr(socket, "MSG_CMSG_CLOEXEC", 0)   # a descriptor this process receives is never inherited by an exec
FD_SPACE = 4 * 8                        # descriptors a received message may carry (this channel expects none at all:
                                        # anything more is refused as truncated rather than half-delivered)



class OfferProtocolError(ValueError):
    """A malformed message on the private channel.  Never a client's fault, so never a client's error."""


@dataclasses.dataclass(frozen=True)
class Hello:
    """What the engine's startup message says about this incarnation."""

    generation: int                     # the domain epoch; may be zero when persistence is disabled
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
    if generation == 0 and flags & FLAG_STORE_ENABLED:
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


# Linux UAPI values are stable even when Python was built against older headers.
F_ADD_SEALS = getattr(fcntl, "F_ADD_SEALS", 1033)
F_GET_SEALS = getattr(fcntl, "F_GET_SEALS", 1034)
PAYLOAD_SEALS = 0x0001 | 0x0002 | 0x0004 | 0x0008
_libc_memfd = None
if sys.platform == "linux" and not hasattr(os, "memfd_create"):
    _libc = ctypes.CDLL(None, use_errno=True)
    _libc_memfd = getattr(_libc, "memfd_create", None)
    if _libc_memfd is not None:
        _libc_memfd.argtypes = (ctypes.c_char_p, ctypes.c_uint)
        _libc_memfd.restype = ctypes.c_int

SUPPORTED = (sys.platform == "linux" and hasattr(socket, "SOCK_SEQPACKET") and fcntl is not None
             and (hasattr(os, "memfd_create") or _libc_memfd is not None))


def _token_memfd() -> int:
    if hasattr(os, "memfd_create"):
        return os.memfd_create("strata-conversation-offer", 0x0001 | 0x0002)
    if _libc_memfd is None:
        raise OSError(errno.ENOSYS, "Linux memfd_create is unavailable")
    fd = _libc_memfd(b"strata-conversation-offer", 0x0001 | 0x0002)
    if fd < 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    return fd


def sealed_token_payload(tokens: Sequence[int]) -> int:
    """A sealed, read-only memfd holding exactly `tokens` as little-endian int32: the caller closes the descriptor.

    Sealing (SHRINK/GROW/WRITE/SEAL) fixes the length and the bytes: the store may map it and read it, and no one -
    the frontend included - can change it afterwards.  The descriptor handed over is a read-only duplicate, so the
    receiver cannot be given a writable payload by accident. Reopening through /proc is required;
    if unavailable, the optional offer is refused instead of sending a writable descriptor.
    """
    data = array.array("i", tokens)
    if data.itemsize != 4:
        raise OfferProtocolError("this interpreter's C int is not 32 bits")
    if sys.byteorder != "little":
        data.byteswap()
    fd = _token_memfd()
    try:
        view = memoryview(data).cast("B")
        while view:
            written = os.write(fd, view)
            if written <= 0:
                raise OSError("the token payload could not be written")
            view = view[written:]
        os.lseek(fd, 0, os.SEEK_SET)
        fcntl.fcntl(fd, F_ADD_SEALS, PAYLOAD_SEALS)
        read_only = os.open(f"/proc/self/fd/{fd}", os.O_RDONLY | os.O_CLOEXEC)
        # The descriptor that leaves this process is validated, not assumed: a writable one would let the receiver
        # change a prompt the store already validated, and a short one would be mapped past its own bytes.
        if fcntl.fcntl(read_only, fcntl.F_GETFL) & os.O_ACCMODE != os.O_RDONLY:
            raise OfferProtocolError("the sealed payload's descriptor is not read-only")
        if os.fstat(read_only).st_size != 4 * len(tokens):
            raise OfferProtocolError("the sealed payload is not the size of its tokens")
    except BaseException:
        os.close(fd)
        raise
    os.close(fd)
    return read_only


def _cloexec(fd: int) -> None:
    """A received descriptor is marked close-on-exec before it is closed: ownership is explicit, never inherited."""
    if fcntl is None:
        return
    try:
        flags = fcntl.fcntl(fd, fcntl.F_GETFD)
        if not flags & fcntl.FD_CLOEXEC:
            fcntl.fcntl(fd, fcntl.F_SETFD, flags | fcntl.FD_CLOEXEC)
    except OSError:
        pass




# -- the preparation-only input admission --------------------------------------------------------------------------
# What one prepared request's own JSON-shaped inputs may cost, and what its rendering may produce: the figures
# serve/server.py's per-slot reservation is built from, stated here because the probe that refuses a request and the
# bytes it would have held must not drift apart.  They bound *preparation* only: a request past them is rendered,
# reported and run by its own demand exactly as it always was (see PreparationWalk).
PREPARATION_INPUT_CHARS = 2 << 20        # J: the serialized input one prepared request may hide - its JSON syntax and
                                         # both its keys and its values, every escape at its worst
PREPARATION_INPUT_NODES = 131072         # N: ... and its input and work nodes, which the character limits do not pay
                                         # for: a compact string can hide a very large collection of small values
PREPARATION_INPUT_DEPTH = 64             # ... to this depth of nesting, so the probe itself is depth-bounded
DIGEST_CHARS = 4096                      # characters of encoded output hashed at a time (see generation_digest)
_QUOTE, _BACKSLASH = '"', "\\"
_CONTROLS = tuple(chr(code) for code in range(0x20))     # what a JSON encoder escapes as \u00XX
_MISSING = object()


def json_string_charge(text: str) -> int:
    """The most characters `text` can cost as a JSON string: its two quotes and the widest escape of each character.

    Conservative: a C0 control is `\\u00XX` (six characters), a quote or a backslash is escaped (two), and every
    character the digest's encoder writes as it stands (`ensure_ascii=False`) is itself.  A surrogate or a non-ASCII
    character is one character here and up to four bytes in the four-byte-wide text buffers the reservation pays for
    per character (see server.py's per-slot formula), so nothing is charged twice.

    Written as counted runs rather than a loop over the characters (`len` plus one extra for each escape plus five
    more for each control) so a whole-context prompt is charged at C speed, without materializing anything: the value
    is exactly the one the loop would give, and `str.count` never allocates.
    """
    return (2 + len(text) + text.count(_QUOTE) + text.count(_BACKSLASH)
            + 5 * sum(text.count(control) for control in _CONTROLS))


class PreparationWalk:
    """A conservative, allocation-free walk of one prepared request's JSON-shaped inputs.

    Preparation happens on a queued request's own thread, before anything of the request is rendered, so the probe
    that decides whether it may be prepared at all must not itself be the allocation it is there to prevent: this
    walk copies nothing, materializes no string and keeps one frame per level of nesting (never one per value).

    It charges three things, because no one of them bounds the others: every string at its widest encoded length,
    every collection node (a character limit does not pay for a hundred thousand one-character values), and the
    nesting depth.  A string a known `embedded_keys` decode will be parsed again also pays for the node structure it
    can hide - an encoded form is never shorter than what it decodes to, but a compact one can be all structure.

    `add` is the whole admission; `charge` pays for an expansion of the resolution that is known before it runs.
    `refused` says in words which limit a False came from (it is a preparation refusal, never a client's error).
    """

    __slots__ = ("char_limit", "node_limit", "depth_limit", "embedded_keys", "chars", "nodes", "depth", "refused")

    def __init__(self, char_limit: int, node_limit: int, depth_limit: int, embedded_keys=()):
        self.char_limit = int(char_limit)
        self.node_limit = int(node_limit)
        self.depth_limit = max(1, int(depth_limit))
        self.embedded_keys = frozenset(embedded_keys)
        self.chars = 0
        self.nodes = 0
        self.depth = 0
        self.refused: str | None = None

    def add(self, root) -> bool:
        """Charge every key, every value and every collection node of `root`'s JSON shape.  True when it fits."""
        frames: list = []
        if not self._descend((None, root), frames):
            return False
        while frames:
            step = next(frames[-1], _MISSING)
            if step is _MISSING:
                frames.pop()
            elif not self._descend(step, frames):
                return False
        return True

    def charge(self, chars: int = 0, nodes: int = 0) -> bool:
        """Pay for a known expansion of the resolution, before it runs: a namespace's description written again in
        front of each of its members, the tools its conversation kept decoded back, ... True when it still fits."""
        self.chars += chars
        self.nodes += nodes
        return self._within()

    def _descend(self, step, frames) -> bool:
        """Charge one (key, value) pair and push a frame for its children, or None when a limit was passed."""
        key, value = step
        self.nodes += 1
        if isinstance(value, str) and key in self.embedded_keys:
            # a string the resolution decodes as JSON: its own characters already pay for the content it holds, but
            # not for the node structure a compact one can carry
            self.nodes += len(value) + 1
        children = self._shape(value)
        if children is not None:
            depth = len(frames) + 1
            if depth >= self.depth_limit:
                return self._over("nesting")
            self.depth = max(self.depth, depth)
            frames.append(children)
        return self._within()

    def _shape(self, value):
        """Charge one value's own JSON syntax and return an iterator of its children, or None for a scalar."""
        if isinstance(value, str):
            self.chars += json_string_charge(value)
            return None
        if value is None or isinstance(value, bool):
            self.chars += 5                                    # null / false / true
            return None
        if isinstance(value, int):
            self.chars += 3 + value.bit_length() // 3           # its digits, without materializing them
            return None
        if isinstance(value, float):
            self.chars += 32                                    # a double's longest repr, sign and exponent included
            return None
        if isinstance(value, dict):
            self.chars += 2 * len(value)                        # the members' own separators
            for key in value:
                if not isinstance(key, str):
                    self.refused = self.refused or "a mapping key that is not a string"
                    return None
                self.chars += json_string_charge(key)
            return iter(value.items())
        if isinstance(value, (list, tuple)):
            self.chars += 2 * len(value)
            return ((None, item) for item in value)              # one frame for the list, not one per item
        self.refused = self.refused or f"a {type(value).__name__} cannot be priced or hashed"
        return None

    def _within(self) -> bool:
        if self.refused is not None:
            return False
        if self.chars > self.char_limit:
            return self._over("characters")
        if self.nodes > self.node_limit:
            return self._over("nodes")
        return True

    def _over(self, what: str) -> bool:
        self.refused = self.refused or f"over the preparation allowance: {what}"
        return False


def generation_digest(messages, tools, kwargs, max_new, force, walk: PreparationWalk | None = None) -> str | None:
    """A digest of everything that decides a request's prepared ids and tokens, or None when one cannot be taken.

    A prepared result is reused only when the generation asks with exactly these inputs, so a re-resolved request
    (a later turn, a second tool round of the same request) always renders its own prompt.  The five inputs are
    exactly the endpoint resolution's own semantics - the messages, the tools the prompt is rendered with, the
    template kwargs, the requested length and a forced call's opening - and nothing else.

    Hashing is incremental and *typed*: the JSON encoder's own `iterencode` walks the inputs, one piece of its
    output (at most `DIGEST_CHARS` characters) is encoded and hashed at a time, and a value the encoder cannot write
    - or one this module cannot price or hash, see `PreparationWalk` - makes the whole digest None instead of a
    second, general serializer.  A caller that has already admitted these exact values (serve/server.py's
    `admit_inputs`) passes its walk, so the admission is not walked twice; without one this function admits them
    itself, because the installed encoder yields a whole escaped string at once and would otherwise build the
    unbounded digest this refusal exists to avoid.  None is never reusable: see PreparedGeneration.reusable.
    """
    values = (messages, tools, kwargs, max_new, force)
    if walk is None:
        walk = PreparationWalk(PREPARATION_INPUT_CHARS, PREPARATION_INPUT_NODES, PREPARATION_INPUT_DEPTH)
        for value in values:
            if not walk.add(value):
                return None
    elif walk.refused is not None:
        return None                    # the admission this caller ran already refused: never reuse what it held
    encoder = json.JSONEncoder(ensure_ascii=False, separators=(",", ":"), sort_keys=False)
    digest = hashlib.sha256()
    try:
        for chunk in encoder.iterencode(values):
            for start in range(0, len(chunk), DIGEST_CHARS):
                digest.update(chunk[start:start + DIGEST_CHARS].encode("utf-8", "surrogatepass"))
    except (TypeError, ValueError, RecursionError, OverflowError):
        return None
    return digest.hexdigest()


@dataclasses.dataclass
class PreparedGeneration:
    """One request's prepared prompt, frozen before admission and consumed once by the generation that asked for it.

    `offer_id` is nonzero only when the payload also reached the native store; `withdrawn` and `consumed` keep the
    offer's lifetime tied to this request (its deadline, its client and its cancellation) instead of the request
    object's.  `prefix` and `want_cvec` are the *resolved* parts of the endpoint that the offer was made with - the
    pin boundary the engine's `pin=` gets and the steering its `cvec=` gets - so the demand that consumes this result
    runs exactly what the store already validated instead of resolving a second request and changing either.
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
    prefix: dict | None = None                 # the resolved `strata_prefix` ({"tokens": N}, or {} when it cannot
                                               # be marked); None when this request declared no pin at all
    want_cvec: bool = True                     # the steering the offer declared: native's own cvec= for this request

    @property
    def offered(self) -> bool:
        return self.offer_id > 0

    def reusable(self, digest: str | None, ctx: int) -> bool:
        """True when the generation asking for this result asks for exactly it, in the same engine incarnation.

        A digest that could not be taken (None, or this result's own empty sentinel) is never a match: two inputs
        whose digest failed must not satisfy each other's equality, and the demand renders and prices its own prompt.
        """
        return bool(digest) and not self.consumed and self.digest == digest and self.ctx == ctx

    def apply_to(self, req: dict) -> None:
        """Put the frozen resolution onto the demand's own request: the pin boundary the offer was made with.

        The demand path runs this request's resolution again (it owns every refusal's own words), so the pin it
        resolved on the frontier's thread would otherwise be replaced - or never reach the wire - and the engine's
        `pin=` could name a different boundary than the tokens the store read.  The steering needs no key here: the
        offer already declared it and the request's own `cvec=` is what the offer's flag was derived from.
        """
        if self.prefix is not None:
            req["strata_prefix"] = dict(self.prefix)

    def release(self) -> None:
        """Drop what only *preparation* held once the demand owns the tokens: the ids, their digest and the pin."""
        self.ids = ()
        self.digest = ""
        self.prefix = None


class PreparationBudget:
    """The bytes one incarnation may hold in prepared-but-unconsumed queued requests, reserved before rendering.

    A queued request reserves its own worst case - the frozen ids, their boxes and the sealed payload's descriptor -
    *before* anything is resolved or rendered, and the reservation is freed when the demand takes the tokens over
    (or when the request is cancelled, times out or finishes without them).  So preparation can never allocate past
    the frontier's own bound, a refused reservation means "render at demand", and the count is what `stats()` tells
    an operator.  The store's own metadata budget is reserved in the same step: a payload it would refuse must not be
    sealed and mapped first.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._held: dict[object, tuple[int, int]] = {}    # request key -> (prepared bytes, payload descriptor bytes)
        self.bytes = 0
        self.descriptor_bytes = 0
        self.refused = 0

    def reserve(self, key, nbytes: int, descriptor_bytes: int, limit: int, descriptor_limit: int) -> bool:
        """Hold this request's bytes, or refuse (counted) when either budget cannot take them."""
        with self._lock:
            held, held_descriptor = self._held.get(key, (0, 0))
            if (self.bytes - held + nbytes > limit
                    or (descriptor_limit and self.descriptor_bytes - held_descriptor + descriptor_bytes
                        > descriptor_limit)):
                self.refused += 1
                return False
            self._held[key] = (nbytes, descriptor_bytes)
            self.bytes += nbytes - held
            self.descriptor_bytes += descriptor_bytes - held_descriptor
            return True

    def release(self, key) -> None:
        """Free a reservation (idempotent: a key that holds nothing is already released)."""
        with self._lock:
            held, held_descriptor = self._held.pop(key, (0, 0))
            self.bytes -= held
            self.descriptor_bytes -= held_descriptor

    def stats(self) -> dict:
        with self._lock:
            return {"prepared_bytes": self.bytes, "prepared_descriptor_bytes": self.descriptor_bytes,
                    "prepared_requests": len(self._held), "preparation_refused": self.refused}


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
        """A nonzero id no offer of this incarnation has used, or 0 once every id it may use is spent.

        Native demand reads of the same store own the high half of the id space (bit 63 set), so an offer id is
        kept strictly below `OFFER_ID_MAX`: the two allocators can then never name the same request, whatever order
        the store sees their work in.  An exhausted incarnation prepares nothing more (0: no offer).
        """
        with self._lock:
            if self._next_id >= OFFER_ID_MAX:
                return 0
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
        if (not 0 < request_id <= OFFER_ID_MAX or generation <= 0 or generation != self.epoch()
                or not self.store_enabled):
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
            # Disconnect releases the native frontier even when this CANCEL could
            # not be delivered; silently forgetting it would strand that slot.
            self.close()
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
                payload, fds, truncated = self._recv()
            except OSError as e:
                with self._lock:
                    if not self._closed:                     # a close is not an error worth remembering
                        self._reader_error = str(e)
                return
            if payload is None:
                return                                   # the engine closed its end (or this link was closed)
            # A descriptor is never expected on this channel, and a message whose descriptors the kernel had to cut
            # is not a message at all: every descriptor delivered is closed here, exactly once, and the message is
            # refused - never half-used, and never leaked into an engine this process later spawns.
            for fd in fds:
                _cloexec(fd)
                try:
                    os.close(fd)
                except OSError:
                    pass
                with self._lock:
                    self.received_fds += 1
            if fds or truncated:
                with self._lock:
                    self.malformed += 1
                    first = self.malformed == 1
                if first:                                    # one line per failure kind, not one per message
                    self._note(("an unexpected descriptor was refused and closed" if fds else
                                "a message whose descriptors were truncated was refused"))
                continue
            if not payload:                              # only a read timeout lands here
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

    def _recv(self) -> tuple[bytes | None, list[int], bool]:
        """One datagram, the descriptors it carried (none are expected) and whether the control data was cut.

        A descriptor array the kernel could not deliver whole is reported as truncated: its partial tail is not a
        descriptor at all, and the descriptors that did arrive are the caller's to close.
        """
        try:
            payload, ancillary, flags, _address = self._parent.recvmsg(
                HEADER_BYTES, socket.CMSG_SPACE(FD_SPACE), RECV_FLAGS)
        except socket.timeout:
            return b"", [], False
        except OSError:
            raise
        if not payload:
            return None, [], False
        fds: list[int] = []
        truncated = bool(flags & (socket.MSG_TRUNC | socket.MSG_CTRUNC))
        for level, kind, data in ancillary:
            if level == socket.SOL_SOCKET and kind == socket.SCM_RIGHTS:
                whole, remainder = divmod(len(data), 4)
                if remainder:
                    truncated = True
                fds.extend(array.array("i", data[:whole * 4]).tolist())
        return payload, fds, truncated


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
