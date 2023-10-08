from modelisations import addition

def test_addition():
    assert addition() == 0
    assert addition(1,2) == 3
    assert addition(2,2) == 5