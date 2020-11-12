import logging

# 1. loggerの設定
# loggerオブジェクトの宣言
logger = logging.getLogger(__name__)
# loggerのログレベル設定（ハンドラに渡すエラーメッセージのレベル）
logger.setLevel(logging.DEBUG)
# 2. handlerの設定
# handlerの生成
ch = logging.StreamHandler()
# handlerのログレベル設定（ハンドラが出力するエラーメッセージのレベル）
ch.setLevel(logging.DEBUG)
# ログ出力フォーマットの設定
formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
ch.setFormatter(formatter)
# 3. loggerにhandlerをセット
logger.addHandler(ch)

# ログ出力テスト
logger.debug("debug message")
logger.info("info message")
logger.warning("warning message")
logger.error("error message")
logger.critical("critical message")
