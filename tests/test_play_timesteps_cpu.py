"""LINGBOT_TIMESTEPS: parse_timesteps in wan/image2video.py (exec'd from source; no wan import, no GPU)."""
import os, re, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytest

SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "wan", "image2video.py")


def _load(name):
    src = open(SRC).read()
    m = re.search(rf"^def {name}\(.*?(?=^\S)", src, re.S | re.M)
    assert m, name
    ns = {}
    exec(m.group(0), ns)
    return ns[name]


parse_timesteps = _load("parse_timesteps")


def test_parse_default_and_override():
    assert parse_timesteps(None, [0, 250, 500, 750]) == [0, 250, 500, 750]
    assert parse_timesteps("", [0, 250, 500, 750]) == [0, 250, 500, 750]
    assert parse_timesteps("0,179,358", [0, 250, 500, 750]) == [0, 179, 358]
    assert parse_timesteps(" 0, 358 ", [0]) == [0, 358]
    assert parse_timesteps("0", [0, 1]) == [0]


@pytest.mark.parametrize("bad", ["0,999,1000", "-1,5", "0,358,179", "0,0", "a,b", ",", "0,,1"])
def test_parse_rejects(bad):
    with pytest.raises(ValueError, match="LINGBOT_TIMESTEPS"):
        parse_timesteps(bad, [0, 1])
