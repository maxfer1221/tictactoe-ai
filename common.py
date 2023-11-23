class OccupiedSpaceException(Exception):
    "Raised when the attempted spot to place is already used up"
    pass

class UndefinedDeviceException(Exception):
    "Raised when a device is not defined (required for pytorch)"
    pass
