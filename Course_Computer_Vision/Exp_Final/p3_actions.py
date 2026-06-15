# actions.py
"""实际系统操作 — 通过 ctypes 调用 Windows API 模拟按键 + 音量控制"""
import time, ctypes, math, subprocess
from ctypes import wintypes

# ====================================================================
# Windows 键盘模拟（无需 pyautogui）
# ====================================================================
user32 = ctypes.windll.user32

# 虚拟键码
VK_LEFT = 0x25;    VK_RIGHT = 0x27;   VK_UP = 0x26;    VK_DOWN = 0x28
VK_SPACE = 0x20;   VK_RETURN = 0x0D;  VK_ESCAPE = 0x1B; VK_TAB = 0x09
VK_F5 = 0x74;      VK_F2 = 0x71;      VK_DELETE = 0x2E
VK_PRIOR = 0x21;   VK_NEXT = 0x22                              # PageUp / PageDown
VK_HOME = 0x24;    VK_END = 0x23                               # Home / End
VK_B = 0x42;       VK_W = 0x57                                 # B / W
VK_MENU = 0x12                                                       # Alt
VK_CONTROL = 0x11; VK_LWIN = 0x5B;    VK_LSHIFT = 0xA0
VK_VOLUME_UP = 0xAF;   VK_VOLUME_DOWN = 0xAE
VK_MEDIA_PLAY_PAUSE = 0xB3
VK_MEDIA_NEXT = 0xB0;  VK_MEDIA_PREV = 0xB1

KEYEVENTF_KEYUP = 0x0002
KEYEVENTF_EXTENDEDKEY = 0x0001


def _press(vk):
    user32.keybd_event(vk, 0, 0, 0)
    time.sleep(0.02)
    user32.keybd_event(vk, 0, KEYEVENTF_KEYUP, 0)

def _hotkey(*vks):
    for vk in vks:
        user32.keybd_event(vk, 0, 0, 0)
    time.sleep(0.04)
    for vk in reversed(vks):
        user32.keybd_event(vk, 0, KEYEVENTF_KEYUP, 0)

def _hotkey_ext(*vks):
    """带扩展键标记的热键（Win/Alt 组合需要）"""
    for vk in vks:
        user32.keybd_event(vk, 0, KEYEVENTF_EXTENDEDKEY, 0)
    time.sleep(0.04)
    for vk in reversed(vks):
        user32.keybd_event(vk, 0, KEYEVENTF_EXTENDEDKEY | KEYEVENTF_KEYUP, 0)


# ====================================================================
# 音量控制（pycaw，可选）
# ====================================================================
VOLUME_OK = False
_VOL_CTRL = None
_VOL_MIN = -65
_VOL_MAX = 0

try:
    from comtypes import CoInitialize, CoUninitialize
    from pycaw.pycaw import AudioUtilities
    CoInitialize()
    dev = AudioUtilities.GetSpeakers()
    _VOL_CTRL = dev.EndpointVolume
    _VOL_MIN, _VOL_MAX = _VOL_CTRL.GetVolumeRange()[0], _VOL_CTRL.GetVolumeRange()[1]
    VOLUME_OK = True
except Exception:
    pass


def _vol_up():
    if VOLUME_OK and _VOL_CTRL:
        cur = _VOL_CTRL.GetMasterVolumeLevel()
        _VOL_CTRL.SetMasterVolumeLevel(min(_VOL_MAX, cur + 6.0), None)
    else:
        _press(VK_VOLUME_UP)

def _vol_down():
    if VOLUME_OK and _VOL_CTRL:
        cur = _VOL_CTRL.GetMasterVolumeLevel()
        _VOL_CTRL.SetMasterVolumeLevel(max(_VOL_MIN, cur - 6.0), None)
    else:
        _press(VK_VOLUME_DOWN)


# ====================================================================
# 操作函数
# ====================================================================
# PPT
ppt_next   = lambda: _press(VK_RIGHT)
ppt_prev   = lambda: _press(VK_LEFT)
ppt_start  = lambda: _press(VK_F5)
ppt_end    = lambda: _press(VK_ESCAPE)
ppt_black  = lambda: _press(VK_B)
ppt_white  = lambda: _press(VK_W)
ppt_home   = lambda: _press(VK_HOME)
ppt_endpg  = lambda: _press(VK_END)

# 媒体
media_play  = lambda: _press(VK_MEDIA_PLAY_PAUSE)
media_next  = lambda: _press(VK_MEDIA_NEXT)
media_prev  = lambda: _press(VK_MEDIA_PREV)
media_delete = lambda: _press(VK_DELETE)
media_vol_up   = _vol_up
media_vol_down = _vol_down

# 窗口
win_minimize   = lambda: _hotkey_ext(VK_LWIN, VK_DOWN)
win_maximize   = lambda: _hotkey_ext(VK_LWIN, VK_UP)
win_switch     = lambda: _hotkey(VK_MENU, VK_TAB)
win_close      = lambda: _hotkey(VK_MENU, 0x73)  # Alt+F4
win_snap_left  = lambda: _hotkey_ext(VK_LWIN, VK_LEFT)
win_snap_right = lambda: _hotkey_ext(VK_LWIN, VK_RIGHT)
win_show_desktop = lambda: _hotkey_ext(VK_LWIN, 0x44)  # Win+D

# 网页
web_refresh   = lambda: _press(VK_F5)
web_forward   = lambda: _hotkey(VK_MENU, VK_RIGHT)   # Alt+Right
web_back      = lambda: _hotkey(VK_MENU, VK_LEFT)    # Alt+Left
web_new_tab   = lambda: _hotkey(VK_CONTROL, 0x54)    # Ctrl+T
web_close_tab = lambda: _hotkey(VK_CONTROL, 0x57)    # Ctrl+W
web_bookmark  = lambda: _hotkey(VK_CONTROL, 0x44)    # Ctrl+D

# 系统工具
sys_screenshot   = lambda: _hotkey(VK_LWIN, VK_LSHIFT, 0x53)  # Win+Shift+S
sys_search       = lambda: _hotkey_ext(VK_LWIN, 0x53)          # Win+S
sys_run          = lambda: _hotkey_ext(VK_LWIN, 0x52)          # Win+R
sys_settings     = lambda: _hotkey_ext(VK_LWIN, 0x49)          # Win+I
sys_task_view    = lambda: _hotkey_ext(VK_LWIN, VK_TAB)        # Win+Tab

# 文件操作
file_copy        = lambda: _hotkey(VK_CONTROL, 0x43)              # Ctrl+C
file_paste       = lambda: _hotkey(VK_CONTROL, 0x56)              # Ctrl+V
file_cut         = lambda: _hotkey(VK_CONTROL, 0x58)              # Ctrl+X
file_delete      = lambda: _press(VK_DELETE)                      # Delete
file_undo        = lambda: _hotkey(VK_CONTROL, 0x5A)              # Ctrl+Z
file_properties  = lambda: _hotkey(VK_MENU, VK_RETURN)            # Alt+Enter
file_rename      = lambda: _press(VK_F2)                          # F2
file_new_folder  = lambda: _hotkey(VK_CONTROL, VK_LSHIFT, 0x4E)  # Ctrl+Shift+N

# 文本编辑
text_select_all = lambda: _hotkey(VK_CONTROL, 0x41)          # Ctrl+A
text_undo       = lambda: _hotkey(VK_CONTROL, 0x5A)          # Ctrl+Z
text_redo       = lambda: _hotkey(VK_CONTROL, 0x59)          # Ctrl+Y
text_save       = lambda: _hotkey(VK_CONTROL, 0x53)          # Ctrl+S
text_find       = lambda: _hotkey(VK_CONTROL, 0x46)          # Ctrl+F
text_replace    = lambda: _hotkey(VK_CONTROL, 0x48)          # Ctrl+H
text_print      = lambda: _hotkey(VK_CONTROL, 0x50)          # Ctrl+P

# 应用启动器
launch_browser = lambda: subprocess.Popen("start https://www.google.com", shell=True)
launch_notepad = lambda: subprocess.Popen("notepad.exe")
launch_calc = lambda: subprocess.Popen("calc.exe")
launch_control_panel = lambda: subprocess.Popen("control.exe")
launch_explorer = lambda: subprocess.Popen("explorer.exe")
launch_taskmgr = lambda: subprocess.Popen("taskmgr.exe")


# ====================================================================
# 功能映射
# ====================================================================
ACTION_MAP = {
    1: {  # PPT 控制
        1: ("下一页", ppt_next),
        2: ("上一页", ppt_prev),
        3: ("开始放映", ppt_start),
        4: ("结束放映", ppt_end),
        5: ("黑屏", ppt_black),
        6: ("白屏", ppt_white),
        7: ("首页", ppt_home),
        8: ("末页", ppt_endpg),
    },
    2: {  # 媒体播放
        1: ("播放/暂停", media_play),
        2: ("下一首", media_next),
        3: ("上一首", media_prev),
        4: ("删除", media_delete),
        5: ("音量+", media_vol_up),
        6: ("音量-", media_vol_down),
    },
    3: {  # 窗口管理
        1: ("最小化", win_minimize),
        2: ("最大化", win_maximize),
        3: ("切换窗口", win_switch),
        4: ("关闭窗口", win_close),
        5: ("分屏左", win_snap_left),
        6: ("分屏右", win_snap_right),
        7: ("显示桌面", win_show_desktop),
    },
    4: {  # 网页浏览
        1: ("刷新", web_refresh),
        2: ("前进", web_forward),
        3: ("后退", web_back),
        4: ("新标签页", web_new_tab),
        5: ("关闭标签", web_close_tab),
        6: ("收藏书签", web_bookmark),
    },
    5: {  # 系统工具
        1: ("截图", sys_screenshot),
        2: ("搜索", sys_search),
        3: ("运行", sys_run),
        4: ("设置", sys_settings),
        5: ("任务视图", sys_task_view),
    },
    6: {  # 文件操作
        1: ("复制", file_copy),
        2: ("粘贴", file_paste),
        3: ("剪切", file_cut),
        4: ("删除", file_delete),
        5: ("撤销", file_undo),
        6: ("属性", file_properties),
        7: ("重命名", file_rename),
        8: ("新建文件夹", file_new_folder),
    },
    7: {  # 文本编辑
        1: ("全选", text_select_all),
        2: ("撤销", text_undo),
        3: ("重做", text_redo),
        4: ("保存", text_save),
        5: ("查找", text_find),
        6: ("替换", text_replace),
        7: ("打印", text_print),
    },
    8: {  # 应用启动器
        1: ("浏览器", launch_browser),
        2: ("记事本", launch_notepad),
        3: ("计算器", launch_calc),
        4: ("控制面板", launch_control_panel),
        5: ("资源管理器", launch_explorer),
        6: ("任务管理器", launch_taskmgr),
    },
}


def execute(func_id, sub_id):
    """执行指定功能的子手势操作。返回操作名称或 None。"""
    func_map = ACTION_MAP.get(func_id)
    if not func_map:
        return None
    action = func_map.get(sub_id)
    if not action:
        return None
    name, fn = action
    try:
        fn()
    except Exception as e:
        print(f"[动作失败] {name}: {e}")
    return name
