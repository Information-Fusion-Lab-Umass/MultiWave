import torch
class EventData(torch.utils.data.Dataset):
    """ Event stream dataset. """

    # fore_demo, fore_times_ip, fore_values_ip, fore_varis_ip
    def __init__(self, data, data_op, freq=True, hascovs=True):
        """
        Data should be a list of event streams; each event stream is a list of dictionaries;
        each dictionary contains: time_since_start, time_since_last_event, type_event
        """
        self.hascovs = hascovs
        self.freq = freq
        self.labels = data_op
        
        if hascovs:
            self.covs = data[3]
        self.times = data[1]
        self.values = data[2]
        self.feats = data[0]
        if freq:
            self.length = data[0][0].shape[0]
        else:
            self.length = data[0].shape[0]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        """ Each returned element is a list, which represents an event stream """
        if self.freq:
            out = []
            for i in range(len(self.times)):
                out.append((self.times[i][idx], self.values[i][idx], self.feats[i][idx]))
            if self.hascovs:
                out.append(self.covs[idx])
        else:
            if self.hascovs:
                out = (self.times[idx], self.values[idx], self.feats[idx], self.covs[idx])
            else:
                out = (self.times[idx], self.values[idx], self.feats[idx])
        return out, self.labels[idx]
def get_dataloader(data, data_out, batch_size, shuffle=True, freq=True, hascovs=True):
    """ Prepare dataloader. """

    ds = EventData(data, data_out, freq=freq, hascovs=hascovs)
    dl = torch.utils.data.DataLoader(
        ds,
        num_workers=2,
        batch_size=batch_size,
        shuffle=shuffle
    )
    return dl

class RNNData(torch.utils.data.Dataset):
    """ Event stream dataset. """
    def __init__(self, data, data_op, freq=False, regularized=True):
        """
        Data should be a list of event streams; each event stream is a list of dictionaries;
        each dictionary contains: time_since_start, time_since_last_event, type_event
        """
        self.freq = freq
        self.data = data
        self.labels = data_op
        self.regularized = regularized

    def __len__(self):
        if self.freq:
            if self.regularized:
                return self.data[0].shape[0]
            else:
                return self.data[0][0].shape[0]
        else:
            return self.data.shape[0]

    def __getitem__(self, idx):
        """ Each returned element is a list, which represents an event stream """
        if self.freq:
            if self.regularized:
                out = [d[idx, :, :] for d in self.data]
            else:
                out = [[d[idx, :] for d in d_arr] for d_arr in self.data]
            return out, self.labels[idx]
        else:
            return self.data[idx,:,:], self.labels[idx]
def get_RNNdataloader(data, data_out, batch_size, shuffle=True, freq=False, regularized=True):
    """ Prepare dataloader. """

    ds = RNNData(data, data_out, freq=freq, regularized=regularized)
    dl = torch.utils.data.DataLoader(
        ds,
        num_workers=2,
        batch_size=batch_size,
        shuffle=shuffle
    )
    return dl