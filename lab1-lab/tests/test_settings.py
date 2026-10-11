from src.settings import Settings


def test_settings_load():
    settings = Settings()

    assert settings.ENVIRONMENT == "test"
    assert settings.APP_NAME == "test_app"
    assert settings.OPENAI_API_KEY == "fake-api-key"
