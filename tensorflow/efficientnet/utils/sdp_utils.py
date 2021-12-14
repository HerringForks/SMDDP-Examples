import smdistributed.dataparallel.tensorflow as sdp

__all__ = [
    'is_using_sdp',
]

def is_using_sdp():
    return sdp.size() > 1
