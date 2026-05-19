"""Receive the DCL chunked-JPEG vision stream over UDP, run the gate-corner
CNN on each reassembled frame, and display the feed with keypoints overlaid.

Packet layout (per spec VADR-TS-002, 24-byte little-endian header):
    frame_id      uint32   unique id of the image frame
    chunk_id      uint16   index of this packet within the frame
    total_chunks  uint16   number of packets that make up the frame
    jpeg_size     uint32   size of the full reassembled JPEG
    payload_size  uint32   size of the JPEG slice in this packet
    sim_time_ns   uint64   simulation timestamp (ns)
followed by `payload_size` bytes of JPEG data.

Press q or Esc in the window to quit.
"""

import socket
import struct
import numpy as np
import cv2
from ultralytics import YOLO

WEIGHTS = r"runs\pose\runs\gate-pose\weights\best.pt"
UDP_IP = "0.0.0.0"      # listen on all interfaces
UDP_PORT = 5600         # spec default
HEADER_FMT = "<IHHIIQ"  # little-endian: u32 u16 u16 u32 u32 u64
HEADER_SIZE = struct.calcsize(HEADER_FMT)   # 24
MAX_PENDING = 8         # how many in-flight frames to track before dropping old ones


def main():
    model = YOLO(WEIGHTS)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 20)
    sock.bind((UDP_IP, UDP_PORT))
    print(f"Listening for DCL vision stream on {UDP_IP}:{UDP_PORT} ...")

    # frame_id -> {"total": int, "chunks": {chunk_id: bytes}}
    pending = {}
    last_shown = -1

    while True:
        packet, _ = sock.recvfrom(65535)
        if len(packet) < HEADER_SIZE:
            continue

        frame_id, chunk_id, total_chunks, jpeg_size, payload_size, _t = \
            struct.unpack(HEADER_FMT, packet[:HEADER_SIZE])
        payload = packet[HEADER_SIZE:HEADER_SIZE + payload_size]

        # ignore frames we've already displayed (stale/out-of-order)
        if frame_id <= last_shown:
            continue

        entry = pending.setdefault(frame_id, {"total": total_chunks, "chunks": {}})
        entry["chunks"][chunk_id] = payload

        # frame complete?
        if len(entry["chunks"]) < entry["total"]:
            continue

        jpeg = b"".join(entry["chunks"][i] for i in range(entry["total"]))
        del pending[frame_id]
        last_shown = frame_id

        # bound memory: drop any older incomplete frames (lost packets)
        for fid in [f for f in pending if f < frame_id]:
            del pending[fid]
        while len(pending) > MAX_PENDING:
            del pending[min(pending)]

        # decode JPEG -> BGR image
        img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            continue  # corrupt frame (e.g. missing chunk content)

        # run the CNN and draw boxes + corner keypoints
        result = model.predict(source=img, verbose=False)[0]
        annotated = result.plot(line_width=2)

        cv2.imshow("DCL vision + gate keypoints", annotated)
        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
            break

    sock.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
