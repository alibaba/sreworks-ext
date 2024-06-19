
class Processer:
    def __init__(self, name='module0'):

        self.name = name
    
    def anomaly_detection(self, data):
        pass

    def parse_timewindow(self, time_window):
        time_window = time_window.split(' ')
        res = {'days':0, 'hours':0, 'minutes':0, 'seconds':0}
        res.update(time_window[0], int(time_window[1]))
        return res


