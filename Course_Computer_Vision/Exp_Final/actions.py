# actions.py
"""实际系统操作 — 通过 ctypes 调用 Windows API 模拟按键 + 音量控制"""
import time, ctypes, math
from ctypes import wintypes

# ====================================================================
# Windows 键盘模拟（无需 pyautogui）
# ====================================================================
user32 = ctypes.windll.user32

# 虚拟键码
VK_LEFT = 0x25;    VK_RIGHT = 0x27;   VK_UP = 0x26;    VK_DOWN = 0x28
VK_SPACE = 0x20;   VK_RETURN = 0x0D;  VK_ESCAPE = 0x1B; VK_TAB = 0x09
VK_F5 = 0x74;      VK_F2 = 0x71;      VK_DELETE = 0x2E
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

# 媒体
media_play  = lambda: _press(VK_MEDIA_PLAY_PAUSE)
media_next  = lambda: _press(VK_MEDIA_NEXT)
media_prev  = lambda: _press(VK_MEDIA_PREV)
media_vol_up   = _vol_up
media_vol_down = _vol_down

# 窗口
win_switch     = lambda: _hotkey(VK_MENU, VK_TAB)
win_minimize   = lambda: _hotkey_ext(VK_LWIN, VK_DOWN)
win_close      = lambda: _hotkey(VK_MENU, 0x73)  # Alt+F4
win_snap_left  = lambda: _hotkey_ext(VK_LWIN, VK_LEFT)
win_snap_right = lambda: _hotkey_ext(VK_LWIN, VK_RIGHT)

# 浏览器
web_scroll_down = lambda: _hotkey(VK_CONTROL, VK_DOWN)   # 近似
web_scroll_up   = lambda: _hotkey(VK_CONTROL, VK_UP)
web_new_tab     = lambda: _hotkey(VK_CONTROL, 0x54)      # Ctrl+T
web_close_tab   = lambda: _hotkey(VK_CONTROL, 0x57)      # Ctrl+W
web_refresh     = lambda: _press(VK_F5)

# 系统
sys_lock         = lambda: _hotkey_ext(VK_LWIN, 0x4C)        # Win+L
sys_screenshot   = lambda: _hotkey(VK_LWIN, VK_LSHIFT, 0x53)  # Win+Shift+S
sys_task_view    = lambda: _hotkey_ext(VK_LWIN, VK_TAB)       # Win+Tab
sys_show_desktop = lambda: _hotkey_ext(VK_LWIN, 0x44)         # Win+D

# 文件快捷操作
file_new_folder  = lambda: _hotkey(VK_CONTROL, VK_LSHIFT, 0x4E)  # Ctrl+Shift+N
file_copy        = lambda: _hotkey(VK_CONTROL, 0x43)              # Ctrl+C
file_paste       = lambda: _hotkey(VK_CONTROL, 0x56)              # Ctrl+V
file_delete      = lambda: _press(VK_DELETE)                      # Delete
file_rename      = lambda: _press(VK_F2)                          # F2

# 输入辅助
input_select_all = lambda: _hotkey(VK_CONTROL, 0x41)          # Ctrl+A
input_undo       = lambda: _hotkey(VK_CONTROL, 0x5A)          # Ctrl+Z
input_save       = lambda: _hotkey(VK_CONTROL, 0x53)          # Ctrl+S
input_find       = lambda: _hotkey(VK_CONTROL, 0x46)          # Ctrl+F
input_ime_switch = lambda: _hotkey_ext(VK_LWIN, VK_SPACE)     # Win+Space


# ====================================================================
# 功能映射
# ====================================================================
ACTION_MAP = {
    1: {  # PPT 控制
        1: ("下一页", ppt_next),
        2: ("上一页", ppt_prev),
        3: ("开始放映", ppt_start),
        4: ("结束放映", ppt_end),
    },
    2: {  # 媒体播放
        1: ("播放/暂停", media_play),
        2: ("下一首", media_next),
        3: ("上一首", media_prev),
        4: ("音量+", media_vol_up),
        5: ("音量-", media_vol_down),
    },
    3: {  # 窗口管理
        1: ("切换窗口", win_switch),
        2: ("最小化", win_minimize),
        3: ("关闭窗口", win_close),
        4: ("分屏左", win_snap_left),
        5: ("分屏右", win_snap_right),
    },
    4: {  # 网页浏览
        1: ("向下滚动", web_scroll_down),
        2: ("向上滚动", web_scroll_up),
        3: ("新标签页", web_new_tab),
        4: ("关闭标签", web_close_tab),
        5: ("刷新页面", web_refresh),
    },
    5: {  # 系统控制
        1: ("锁屏", sys_lock),
        2: ("截图", sys_screenshot),
        3: ("任务视图", sys_task_view),
        4: ("显示桌面", sys_show_desktop),
    },
    6: {  # 文件快捷操作
        1: ("新建文件夹", file_new_folder),
        2: ("复制", file_copy),
        3: ("粘贴", file_paste),
        4: ("删除", file_delete),
        5: ("重命名", file_rename),
    },
    7: {  # 输入辅助
        1: ("全选", input_select_all),
        2: ("撤销", input_undo),
        3: ("保存", input_save),
        4: ("查找", input_find),
        5: ("切换输入法", input_ime_switch),
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
