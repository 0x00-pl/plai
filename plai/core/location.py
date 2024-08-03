
class Location:
    pass


class DummyLocation(Location):
    def __str__(self):
        return "DummyLocation"


class NamedLocation(Location):
    def __init__(self, name: str):
        self.name = name

    def __str__(self):
        return self.name


class LocationFileLine(Location):
    def __init__(self, file: str, line: int):
        self.file = file
        self.line = line

    def __str__(self):
        return f"{self.file}: {self.line}"


def get_variable_name(location: Location) -> str:
    if isinstance(location, NamedLocation):
        return location.name
    else:
        raise ValueError(f"Unknown location type: {location}")
