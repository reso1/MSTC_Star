import math
import bisect
import networkx as nx

V0 = 1.0  # m/s
SQRT_2 = math.sqrt(2)
PI = math.pi
YAW = {(1, 0): ('E', 0), (1, 1): ('NE', PI/4), (0, 1): ('N', PI/2),
       (-1, 1): ('NW', 3*PI/4), (-1, 0): ('W', PI), (-1, -1): ('SW', 5*PI/4),
       (0, -1): ('S', 3*PI/2), (1, -1): ('SE', 7*PI/4)}


class State:
    def __init__(self, x=0.0, y=0.0, yaw=0.0, ts=0.0):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.ts = ts


class Robot:
    def __init__(self, path, G):
        self.state = path[0]
        self.S = path
        self.N = len(self.S)
        self.T, self.V = self.__init_time_series(G)

    def get_cur_state(self, ts):
        """ locate ts in time series T using binary search """
        if ts >= self.T[-1]:
            self.state.ts = ts
            return self.N-1, self.state

        ti = bisect.bisect(self.T, ts) - 1
        dx = self.S[ti+1][0] - self.S[ti][0]
        dy = self.S[ti+1][1] - self.S[ti][1]
        k = (ts - self.T[ti]) / (self.T[ti+1] - self.T[ti])
        x_sign = 0 if dx == 0 else (1 if dx > 0 else -1)
        y_sign = 0 if dy == 0 else (1 if dy > 0 else -1)

        self.state = State(
            x=self.S[ti][0] + k*dx, y=self.S[ti][1] + k*dy,
            yaw=YAW[(x_sign, y_sign)][1], ts=ts)

        return ti, self.state

    def __init_time_series(self, G: nx.Graph):
        T, V = [0.0] * self.N, [0.0] * (self.N-1)

        cur = self.state
        for i, nxt in enumerate(self.S[1:]):
            dist = 1.0 if cur[0] == nxt[0] or cur[1] == nxt[1] else SQRT_2
            V[i] = V0
            T[i+1] = T[i] + dist / V[i]
            cur = nxt

        return T, V
