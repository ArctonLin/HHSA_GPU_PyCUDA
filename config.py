import os

class Config:
    def __init__(self, profile_path="HHSA_profile.txt"):
        self.mode = 0
        self.ensemble = 0
        self.TargetSTDEV = 0.0
        self.dt = 0.0
        self.Time_cell = 0
        self.Freq_cell = 0
        self.Holo_Freq_cell = 0
        self.ImageSizeX1 = 0
        self.ImageSizeY1 = 0
        self.ImageSizeX2 = 0
        self.ImageSizeY2 = 0
        self.Time_cell_group = 0
        self.EMD_type = "CEEMD"
        
        self.load(profile_path)

    def load(self, path):
        with open(path, 'r') as f:
            lines = [l.strip() for l in f.readlines() if l.strip()]
        if len(lines) < 13:
            raise ValueError("Profile file format is incorrect or missing parameters")
        self.mode = int(lines[0])
        self.ensemble = int(lines[1])
        self.TargetSTDEV = float(lines[2])
        self.dt = float(lines[3])
        self.Time_cell = int(lines[4])
        self.Freq_cell = int(lines[5])
        self.Holo_Freq_cell = int(lines[6])
        self.ImageSizeX1 = int(lines[7])
        self.ImageSizeY1 = int(lines[8])
        self.ImageSizeX2 = int(lines[9])
        self.ImageSizeY2 = int(lines[10])
        self.Time_cell_group = int(lines[11])
        self.EMD_type = lines[12]

if __name__ == "__main__":
    cfg = Config()
    print("EMD type:", cfg.EMD_type)
    print("Mode:", cfg.mode)
    print("Ensemble:", cfg.ensemble)
    print("Target STDEV:", cfg.TargetSTDEV)
    print("dt:", cfg.dt)
