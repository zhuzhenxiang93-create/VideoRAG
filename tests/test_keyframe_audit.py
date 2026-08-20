from scripts.audit_keyframes import hamming


def test_hamming_distance():
    assert hamming(0b0000, 0b0000) == 0
    assert hamming(0b0000, 0b1011) == 3
