# 斯诺克画线助手（Chrome 插件）

在虎牙斯诺克直播的视频画面上直接识别球台和球，按一个球的宽度画出瞄准线、进球线和白球走位，并判断路线是否被别的球挡住。

和桌面悬浮窗版相比：插件直接读取网页里视频的原始画面，不需要截屏权限，也不用手动对齐窗口；画线图层自动贴在视频上，窗口缩放、网页全屏和真正的全屏都会跟着走。

## 安装（开发者模式加载）

1. 打开 Chrome，地址栏输入 `chrome://extensions` 回车。
2. 打开右上角的「开发者模式」。
3. 点「加载已解压的扩展程序」，选择这个 `extension` 文件夹。
4. 打开或刷新虎牙的斯诺克直播间，视频左上角会出现「斯诺克画线」状态框。

修改代码后，在 `chrome://extensions` 里点插件卡片上的刷新按钮，再刷新直播页面。

## 使用

| 操作 | 作用 |
| --- | --- |
| 按住 Alt（Mac 上是 Option）点一颗球 | 选为目标球 |
| Alt + R | 取消目标球 |
| Alt + B | 显示 / 隐藏每颗球的识别颜色圈（默认隐藏，用来检查识别对不对） |
| Alt + O | 切换台面朝向（转播主视角 / 俯视全台） |
| Alt + A | 清空已学习的台面视角，重新学习 |
| Alt + H，或点工具栏上的插件图标 | 显示 / 隐藏画线 |
| Alt + S | 保存当前画面和识别结果到下载文件夹（画错时用来反馈问题，最好先暂停视频） |

- 镜头切到能看到整张台的机位、稳定约半秒后，自动学会这个视角（最多记 4 个）；拍球员、特写、回放时自动隐藏画线。
- 球在滚动时不画线，停稳后才计算。
- 轨迹带被别的球挡住时标红，并圈出挡路的球；会优先选不被挡住、切角最小的袋口。
- 暂停视频时也会继续识别，可以暂停下来研究一个球局。

## 结构

```
extension/
  manifest.json        Manifest V3；只在 *.huya.com 上运行
  lib/opencv.js        OpenCV.js 4.9.0（docs.opencv.org 官方构建）
  src/content.js       运行在直播页面：找视频、取帧、画线、处理 Alt+点击和快捷键
  src/engine.html/.js  识别引擎：隐藏的沙盒 iframe（OpenCV.js 需要动态代码，只有沙盒页允许）
  src/vision.js        球台 / 球识别、颜色分类、台面视角学习（对应 Python 版 vision.py）
  src/geometry.js      透视变换、进球线、遮挡判断（对应 geometry.py）
  src/tracking.js      跨帧跟踪球、判断是否停稳（对应 tracking.py）
  src/analyzer.js      每帧的整体流程（对应 analyzer.py）
  src/background.js    工具栏按钮：显示 / 隐藏
  test/run-tests.js    在 Node 里用项目截图测试识别代码
```

识别逻辑与项目根目录的 Python 版逐行对应、阈值一致，两边用同一份 `test_data/ground_truth.json` 测试。

## 测试

在项目根目录运行：

```bash
node extension/test/run-tests.js
```

会用项目里的截图跑模拟直播（切镜头、球员遮挡、球滚动、进球）和降质识别（缩小、变暗、变亮、模糊），和 Python 版的 `test_stream.py` 检查同样的结果。
