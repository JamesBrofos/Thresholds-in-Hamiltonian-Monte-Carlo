class Info:
    def __init__(self):
        self.success: bool = True
        self.invalid: bool = False
        self.logdet = 0.0

class LeapfrogInfo(Info):
    pass

class GeneralizedLeapfrogInfo(Info):
    def __init__(self):
        super().__init__()
        self.num_iters_pos: int = 0
        self.num_iters_mom: int = 0

class ImplicitMidpointInfo(Info):
    def __init__(self):
        super().__init__()
        self.num_iters: int = 0

class CoupledVectorFieldInfo(Info):
    def __init__(self):
        super().__init__()
        self.num_iters: int = 0

class LagrangianLeapfrogInfo(Info):
    pass
