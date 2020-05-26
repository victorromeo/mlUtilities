from ml_utilities.noise_utils import white, pink, blue, violet, brown
import numpy 

def test_white():
    n = 1000
    samples = white(n)
    assert len(samples) == n, f'Samples length expected {n}'
    assert all([ isinstance(x, numpy.float32) for x in samples]), 'Expected values to be floats'
    assert all([(x != 0.0) for x in samples]), 'Expected values to be non-zero'
    
def test_pink():
    n = 1000
    samples = pink(n)
    assert len(samples) == n, f'Samples length expected {n}'
    assert all([ isinstance(x, numpy.float32) for x in samples]), 'Expected values to be floats'
    assert all([(x != 0.0) for x in samples]), 'Expected values to be non-zero'

def test_violet():
    n = 1000
    samples = violet(n)
    assert len(samples) == n, f'Samples length expected {n}'
    assert all([ isinstance(x, numpy.float32) for x in samples]), 'Expected values to be floats'
    assert all([(x != 0.0) for x in samples]), 'Expected values to be non-zero'

def test_blue():
    n = 1000
    samples = blue(n)
    assert len(samples) == n, f'Samples length expected {n}'
    assert all([ isinstance(x, numpy.float32) for x in samples]), 'Expected values to floats'
    assert all([(x != 0.0) for x in samples]), 'Expected values to be non-zero'

def test_brown():
    n = 1000
    samples = brown(n)

    assert len(samples) == n, f'Samples length expected {n}'
    assert all([ isinstance(x, numpy.float32) for x in samples]), 'Expected values to be floats'
    assert all([(x != 0.0) for x in samples]), 'Expected values to be non-zero'