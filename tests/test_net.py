import socket
from threading import Thread

import numpy as np

from mail_bench.net import recv_message, send_message


def test_socket_codec_round_trip_without_pickle():
    sender, receiver = socket.socketpair()
    payload = {
        "kind": "chunk",
        "array": np.arange(21, dtype=np.float32).reshape(3, 7),
        "flags": [True, False],
    }
    thread = Thread(target=send_message, args=(sender, payload))
    thread.start()
    decoded = recv_message(receiver)
    thread.join()
    sender.close()
    receiver.close()
    assert decoded["kind"] == "chunk"
    assert decoded["flags"] == [True, False]
    np.testing.assert_array_equal(decoded["array"], payload["array"])
