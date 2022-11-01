import ctypes
import glob
import os
import platform
import sys


def check_pip_env():
    filter_package_name = 'megbrain'
    need_package_name = 'megengine'

    import pkg_resources

    installed_packages = pkg_resources.working_set
    installed_packages_list = sorted(["%s" % i.key.lower() for i in installed_packages])
    assert (
        filter_package_name not in installed_packages_list
    ), "Your python packages may be corrupted by installing internal&external versions at the same time. To fix it, try to uninstall {} and {}, then install {} again".format(
        filter_package_name, need_package_name, need_package_name
    )

    assert (
        "megenginelite" not in installed_packages_list
    ), "Your python packages may be corrupted by installing internal&external versions at the same time. To fix it, try to uninstall megenginelite and {}, then install {} again".format(
        need_package_name, need_package_name
    )


def check_termux():
    if os.getenv("TERMUX_VERSION"):
        try:
            import cv2
        except Exception as exc:
            print("Run MegEngine python interface at Android/Termux env")
            print("!!!You need build opencv-python manually!!!, by run sh:")
            print(
                "https://github.com/MegEngine/MegEngine/blob/master/scripts/whl/android/android_opencv_python.sh"
            )
            raise exc


def check_windows():
    if sys.platform == "win32":
        lib_path = os.path.join(os.path.dirname(__file__), "core/lib")
        dll_paths = list(filter(os.path.exists, [lib_path,]))
        assert len(dll_paths) > 0

        kernel32 = ctypes.WinDLL("kernel32.dll", use_last_error=True)
        has_load_library_attr = hasattr(kernel32, "AddDllDirectory")
        old_error_mode = kernel32.SetErrorMode(0x0001)

        kernel32.LoadLibraryW.restype = ctypes.c_void_p
        if has_load_library_attr:
            kernel32.AddDllDirectory.restype = ctypes.c_void_p
            kernel32.LoadLibraryExW.restype = ctypes.c_void_p

        for dll_path in dll_paths:
            if sys.version_info >= (3, 8):
                os.add_dll_directory(dll_path)
            elif has_load_library_attr:
                res = kernel32.AddDllDirectory(dll_path)
                if res is None:
                    err = ctypes.WinError(ctypes.get_last_error())
                    err.strerror += ' Error adding "{}" to the DLL search PATH.'.format(
                        dll_path
                    )
                    raise err
            else:
                print("WARN: python or OS env have some issue, may load DLL failed!!!")

        dlls = glob.glob(os.path.join(lib_path, "*.dll"))
        path_patched = False
        for dll in dlls:
            is_loaded = False
            if has_load_library_attr:
                res = kernel32.LoadLibraryExW(dll, None, 0x00001100)
                last_error = ctypes.get_last_error()
                if res is None and last_error != 126:
                    err = ctypes.WinError(last_error)
                    err.strerror += ' Error loading "{}" or one of its dependencies.'.format(
                        dll
                    )
                    err.strerror += " \nplease install VC runtime from: "
                    err.strerror += " \nhttps://docs.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist?view=msvc-160"
                    raise err
                elif res is not None:
                    is_loaded = True
            if not is_loaded:
                if not path_patched:
                    os.environ["PATH"] = ";".join(dll_paths + [os.environ["PATH"]])
                    path_patched = True
                res = kernel32.LoadLibraryW(dll)
                if res is None:
                    err = ctypes.WinError(ctypes.get_last_error())
                    err.strerror += ' Error loading "{}" or one of its dependencies.'.format(
                        dll
                    )
                    err.strerror += " \nplease install VC runtime from: "
                    err.strerror += " \nhttps://docs.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist?view=msvc-160"
                    raise err

        kernel32.SetErrorMode(old_error_mode)


def check_misc():
    check_pip_env()
    check_termux()
    check_windows()
