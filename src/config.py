import yaml
import os

class ConfigSection:
    def __init__(self, d):
        for k, v in d.items():
            if isinstance(v, dict):
                setattr(self, k, ConfigSection(v))
            else:
                setattr(self, k, v)

    def to_dict(self):
        d = {}
        for k, v in self.__dict__.items():
            if isinstance(v, ConfigSection):
                d[k] = v.to_dict()
            else:
                d[k] = v
        return d

class Config:
    def __init__(self, config_dict):
        self.input = ConfigSection(config_dict.get('input', {}))
        self.detector = ConfigSection(config_dict.get('detector', {}))
        self.tracker = ConfigSection(config_dict.get('tracker', {}))
        self.visualization = ConfigSection(config_dict.get('visualization', {}))
        self.output = ConfigSection(config_dict.get('output', {}))
        self.api = ConfigSection(config_dict.get('api', {}))

    @classmethod
    def load(cls, filepath):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Config file not found: {filepath}")
        with open(filepath, 'r') as f:
            d = yaml.safe_load(f)
        return cls(d)

    def update(self, new_dict):
        """Recursively update configuration with values from a dictionary."""
        def update_section(section, data):
            for k, v in data.items():
                if hasattr(section, k):
                    current_val = getattr(section, k)
                    if isinstance(current_val, ConfigSection) and isinstance(v, dict):
                        update_section(current_val, v)
                    else:
                        setattr(section, k, v)
                else:
                    if isinstance(v, dict):
                        setattr(section, k, ConfigSection(v))
                    else:
                        setattr(section, k, v)

        for section_name in ['input', 'detector', 'tracker', 'visualization', 'output', 'api']:
            if section_name in new_dict:
                section = getattr(self, section_name)
                update_section(section, new_dict[section_name])

    def to_dict(self):
        return {
            'input': self.input.to_dict(),
            'detector': self.detector.to_dict(),
            'tracker': self.tracker.to_dict(),
            'visualization': self.visualization.to_dict(),
            'output': self.output.to_dict(),
            'api': self.api.to_dict()
        }
