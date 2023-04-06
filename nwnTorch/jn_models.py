import torch
# import numpy as np

class junction_sydney:
    def __init__(self, N, params):
        self.params = params
        self.V      = torch.zeros(N)
        self.G      = torch.zeros(N)
        self.L      = torch.zeros(N)
        self.switch = torch.zeros(N, dtype = bool)

        self.params["grow"]      = 5     #smax
        self.params["decay"]     = 10    #b
        self.params["collapse"]  = False
        self.params["precision"] = True # to avoid the overshoot of conductance
        self.params["Lcrit"]     = 1e-1
        self.params["Lmax"]      = 1.5e-1

    def updateG(self):
        self.switch      = abs(self.L) >= self.params["Lcrit"]

        phi    = 0.81
        C0     = 10.19
        J1     = 0.0000471307
        A      = 0.17
        d      = (self.params["Lcrit"] - abs(self.L)) \
                * self.params["grow"]/self.params["Lcrit"]
        d[d<0] = 0
        tun    = 2/A * d**2 / phi**0.5 * torch.exp(C0*phi**2 * d)/J1
        
        if self.params["precision"]:
            shift = self.params["Ron"] ** 2 \
                    / (self.params["Roff"] - self.params["Ron"])
        else:
            shift = 0

        self.G = 1/(tun + self.params["Ron"] + shift) \
                            + 1/self.params["Roff"]

    def updateL(self):
        was_open = abs(self.L) >= self.params["Lcrit"]
        
        self.L += (abs(self.V) > self.params["Vset"]) *\
                    (abs(self.V) - self.params["Vset"]) *\
                    torch.sign(self.V) * self.params["dt"]
                    
        self.L -= (self.params["Vreset"] > abs(self.V)) *\
                    (self.params["Vreset"] - abs(self.V)) *\
                    torch.sign(self.L) * self.params["dt"] * self.params["decay"]

        self.L  = torch.clamp(self.L, -self.params["Lmax"], self.params["Lmax"])

        if self.params["collapse"]:
            just_closed = was_open & (abs(self.L) < self.params["Lcrit"])
            
            self.L[just_closed] /= self.params["decay"]

class junction_linear:
    def __init__(self, N, params):
        """
        dw/dt = Roff * S / (beta * Rw) - alpha * w(t)
        Rw    = Roff * (1-w) + w * Ron

        Analytical (mean field):
        dw/dt = 1/beta * (I - chi * Omega * W) ^ (-1) * Omega * S - alpha * omega + (noise)

        Noise is not included atm. 
        """
        self.params = params
        self.V      = torch.zeros(N)
        self.G      = torch.zeros(N)
        self.L      = torch.zeros(N)
        self.switch = torch.zeros(N, dtype = bool)

        self.params["Roff"]  = 1e5
        self.params["Ron"]   = 1e4
        self.params["alpha"] = 1     #decay constant
        self.params["beta"]  = 2e-1  #in VB's code we set this to 1, effective activation voltage per unit time
        self.params["sigma"] = .1    #noise variance
        self.params["chi"]   = (self.params["Roff"] - self.params["Ron"]) / self.params["Roff"]

        # self.params["chi"]   = .1     #non-linearity parameter, between [0,1]

    def updateG(self):
        self.G = 1/(self.params["Roff"] * (1 - self.L) + self.params["Ron"] * self.L)
        # self.G = 1/(self.params["Roff"] * self.L + self.params["Ron"]*(1-self.L))
    def updateL(self):
        self.L += (self.params["Roff"]/self.params["beta"] *\
                    self.V * self.G - self.params["alpha"]*self.L) *\
                    self.params["dt"]


        # self.L = self.params["alpha"] * self.L - self.V * self.G /self.params["beta"]
        self.L = torch.clamp(self.L, 0, 1)

class junction_miranda:
    def __init__(self, N, params):
        """
        From Francesco's preprint.
        Parameters from Miranda et al.'s paper, arbitrary.
        This is a bipolar model (i.e. Activate and deactivate in different polarities).        
        """
        self.params = params
        self.V      = torch.zeros(N)
        self.G      = torch.zeros(N)
        self.L      = torch.zeros(N)
        self.switch = torch.zeros(N, dtype = bool)
        
        # self.params["kappa_P0"] = 1e-15
        # self.params["kappa_D0"] = 1e-6
        # self.params["eta_P0"]   = 16
        # self.params["eta_D0"]   = 30
        self.params["kappa_P0"] = 1e-12
        self.params["kappa_D0"] = 1e-6
        self.params["eta_P0"]   = 32
        self.params["eta_D0"]   = 60

    def updateG(self):
        # self.G = 1/(self.params["Roff"] * (1 - self.L) + self.params["Ron"] * self.L)
        Gmin = 1/self.params["Roff"]
        Gmax = 1/self.params["Ron"]
        self.G = Gmin * (1-self.L) + Gmax * self.L
        # self.G = 1/self.params["Roff"] * self.L + 1/self.params["Ron"] * (1-self.L)

    def updateL(self):
        eta_P   = self.params["kappa_P0"] * torch.exp(self.params["eta_P0"] * self.V)
        eta_D   = self.params["kappa_D0"] * torch.exp(-self.params["eta_D0"] * self.V)
        # self.L += (eta_P * self.V * (1-self.L) - eta_D * self.V * self.L) \
        #              * self.params["dt"]
        self.L += (eta_P * (1-self.L) - eta_D * self.L) * self.params["dt"]
        self.L  = torch.clamp(self.L, 0, 1)


junction_dict = {
                "sydney" : junction_sydney,
                "linear" : junction_linear,
                "miranda": junction_miranda
                }

