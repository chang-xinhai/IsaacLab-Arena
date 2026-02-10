class IsaacLabArenaError(RuntimeError):
    """Base exception for IsaacLab Arena environment errors."""

    def __init__(self, message: str = "IsaacLab Arena error"):
        self.message = message
        super().__init__(self.message)


class IsaacLabArenaConfigError(IsaacLabArenaError):
    """Exception raised for invalid environment configuration."""

    def __init__(self, invalid: list, available: list, key_type: str = "keys"):
        msg = f"Invalid {key_type}: {invalid}. Available: {sorted(available)}"
        super().__init__(msg)
        self.invalid = invalid
        self.available = available


class IsaacLabArenaCameraKeyError(IsaacLabArenaConfigError):
    """Exception raised when camera_keys don't match available cameras."""

    def __init__(self, invalid: list, available: list):
        super().__init__(invalid, available, "camera_keys")


class IsaacLabArenaStateKeyError(IsaacLabArenaConfigError):
    """Exception raised when state_keys don't match available state terms."""

    def __init__(self, invalid: list, available: list):
        super().__init__(invalid, available, "state_keys")