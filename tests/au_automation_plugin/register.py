"""Register a fixture bundle only in this process, without system installation."""

import ctypes as ct
import plistlib
from pathlib import Path

# Components refer to factory functions in loaded bundles for this process's life.
_BUNDLES = []
_REGISTERED_PATHS = set()


def register_bundle(path):
    path = Path(path).resolve()
    if str(path) in _REGISTERED_PATHS:
        return str(path)
    info = plistlib.loads((path / "Contents/Info.plist").read_bytes())
    cf = ct.CDLL("/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation")
    au = ct.CDLL("/System/Library/Frameworks/AudioToolbox.framework/AudioToolbox")
    cf.CFURLCreateFromFileSystemRepresentation.argtypes = [
        ct.c_void_p,
        ct.c_char_p,
        ct.c_long,
        ct.c_bool,
    ]
    cf.CFURLCreateFromFileSystemRepresentation.restype = ct.c_void_p
    cf.CFBundleCreate.argtypes = [ct.c_void_p, ct.c_void_p]
    cf.CFBundleCreate.restype = ct.c_void_p
    cf.CFBundleLoadExecutable.argtypes = [ct.c_void_p]
    cf.CFBundleLoadExecutable.restype = ct.c_bool
    cf.CFStringCreateWithCString.argtypes = [ct.c_void_p, ct.c_char_p, ct.c_uint32]
    cf.CFStringCreateWithCString.restype = ct.c_void_p
    cf.CFBundleGetFunctionPointerForName.argtypes = [ct.c_void_p, ct.c_void_p]
    cf.CFBundleGetFunctionPointerForName.restype = ct.c_void_p
    cf.CFRelease.argtypes = [ct.c_void_p]

    class Description(ct.Structure):
        _fields_ = [
            (name, ct.c_uint32) for name in ["type", "subtype", "manufacturer", "flags", "mask"]
        ]

    au.AudioComponentRegister.argtypes = [
        ct.POINTER(Description),
        ct.c_void_p,
        ct.c_uint32,
        ct.c_void_p,
    ]
    au.AudioComponentRegister.restype = ct.c_void_p
    encoded = str(path).encode()
    url = cf.CFURLCreateFromFileSystemRepresentation(None, encoded, len(encoded), True)
    bundle = cf.CFBundleCreate(None, url)
    cf.CFRelease(url)
    if not bundle or not cf.CFBundleLoadExecutable(bundle):
        raise RuntimeError(f"Cannot load AU fixture bundle {path}")
    _BUNDLES.append((bundle, cf, au))
    for component in info["AudioComponents"]:
        symbol = cf.CFStringCreateWithCString(
            None, component["factoryFunction"].encode(), 0x08000100
        )
        factory = cf.CFBundleGetFunctionPointerForName(bundle, symbol)
        cf.CFRelease(symbol)
        if not factory:
            raise RuntimeError(f"Missing AU fixture factory {component['factoryFunction']}")
        description = Description(
            *[
                int.from_bytes(component[key].encode(), "big")
                for key in ["type", "subtype", "manufacturer"]
            ],
            0,
            0,
        )
        name = cf.CFStringCreateWithCString(None, component["name"].encode(), 0x08000100)
        registered = au.AudioComponentRegister(
            ct.byref(description), name, component["version"], factory
        )
        cf.CFRelease(name)
        if not registered:
            raise RuntimeError(f"Cannot register AU fixture {component['name']}")
    _REGISTERED_PATHS.add(str(path))
    return str(path)
