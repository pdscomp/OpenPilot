"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
from collections.abc import Callable
import pyray as rl

from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.sunnypilot.selfdrive.controls.lib.torque_tune import (TUNE_PARAM_BY_SIZE, jerk_aware_has_effect, label_for,
                                                                      stored_tune_versions, versions_by_label)
from openpilot.system.ui.lib.application import gui_app, FontWeight
from openpilot.system.ui.lib.multilang import tr
from openpilot.system.ui.sunnypilot.lib.utils import NoElideButtonAction
from openpilot.system.ui.sunnypilot.widgets.list_view import ListItemSP, toggle_item_sp, option_item_sp
from openpilot.system.ui.sunnypilot.widgets.tree_dialog import TreeOptionDialog, TreeFolder, TreeNode
from openpilot.system.ui.widgets import Widget, DialogResult
from openpilot.system.ui.widgets.network import NavButton
from openpilot.system.ui.widgets.scroller_tici import Scroller



class TorqueSettingsLayout(Widget):
  def __init__(self, back_btn_callback: Callable):
    super().__init__()
    self._back_button = NavButton(tr("Back"))
    self._back_button.set_click_callback(back_btn_callback)
    self._torque_version_dialog: TreeOptionDialog | None = None
    self.cached_torque_versions = versions_by_label()
    items = self._initialize_items()
    self._scroller = Scroller(items, line_separator=True, spacing=0)

  def _initialize_items(self):
    self._jerk_aware_toggle = toggle_item_sp(
      param="LateralJerkTorqueController",
      title=lambda: tr("Lateral Jerk Torque Controller"),
      description=lambda: tr("Looks ahead at planned steering to reduce sudden corrections, so the wheel moves " +
                             "more smoothly through turns. Works with Self-Tune and custom tuning. " +
                             "Thanks to @twilsonco for the implementation."),
    )
    # one tune per model size; controlsd swaps them as modelV2.big changes
    def tune_row(big: bool, title: str, description: str) -> ListItemSP:
      return ListItemSP(title=title, description=description, action_item=NoElideButtonAction(tr("SELECT")),
                        callback=lambda: self._show_torque_version_dialog(big))
    self._tune_rows = {
      False: tune_row(False, tr("Torque Control Tune Version (Small Models)"),
                      tr("Select the version of Torque Control Tune to use while the small model is driving.")),
      True: tune_row(True, tr("Torque Control Tune Version (Big Models)"),
                     tr("Select the version of Torque Control Tune to use while a big model is driving.")),
    }
    self._self_tune_toggle = toggle_item_sp(
      param="LiveTorqueParamsToggle",
      title=lambda: tr("Self-Tune"),
      description=lambda: tr("Enables self-tune for Torque lateral control for platforms that do not use " +
                             "Torque lateral control by default."),
    )
    self._relaxed_tune_toggle = toggle_item_sp(
      param="LiveTorqueParamsRelaxedToggle",
      title=lambda: tr("Less Restrict Settings for Self-Tune (Beta)"),
      description=lambda: tr("Less strict settings when using Self-Tune. This allows torqued to be more " +
                             "forgiving when learning values."),
    )
    self._speed_dep_toggle = toggle_item_sp(
      param="SpeedDependentTorqueToggle",
      title=lambda: tr("Speed-Dependent Self-Tune (Beta)"),
      description=lambda: tr("Learns separate torque parameters at different speeds. " +
                             "Improves steering at both low and high speeds for supported cars."),
