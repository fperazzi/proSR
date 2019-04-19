
import torch

def range_splits(tensor,split_ranges,dim):
    """Splits the tensor according to chunks of split_ranges.

    Arguments:
        tensor (Tensor): tensor to split.
        split_ranges (list(tuples(int,int))): sizes of chunks (start,end).
        dim (int): dimension along which to split the tensor.
    """
    return tuple(tensor.narrow(int(dim), start, end - start) for start, end in split_ranges)

def max_dimension_split(tensor,max_dimension,padding,dim):
    """Splits the tensor in chunks of max_dimension

    Arguments:
        tensor (Tensor): tensor to split.
        max_dimension (int): maximum allowed size for dim.
        dim (int): dimension along which to split the tensor.
    """
    assert padding < max_dimension
    dimension = tensor.size(dim)
    num_splits = int(dimension / max_dimension) + \
        int(dimension % max_dimension != 0)
    if num_splits == 1: return [tensor]
    else:
        split_ranges = []
        for i in range(num_splits):
            start = max(0,i * max_dimension-padding)
            end   = min(dimension,(i+1) * max_dimension)
            split_ranges.append((start,end))
    return range_splits(tensor,split_ranges,dim)

def cat_chunks(tensors,padding,dim):
    """Concatenate the tensors along the axis dim.

    Arguments:
        tensors (list(Tensor)): list of tensors to concatenate.
        padding (int): padding used to split the tensors.
        dim (int): dimension along which to cat the tensors.
    """
    def remove_padding(tensor,padding,dim):
        tensor = tensor.narrow(dim,padding,tensor.size(dim)-padding)
        return tensor

    return torch.cat([tensors[0].to('cpu').float()]+
                     [remove_padding(t.to('cpu').float(),padding,dim) for t in tensors[1:]],dim=dim)

def chunks_iter(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


class DataChunks(object):
    # Assume chunks have the same size
    """Splits a dict of tensor in chunks of max_dimension and concatenate the output tensor.

    Arguments:
        data (dict): dictionary of tensors.
        max_dimension (int): maximum dimension allowed for a tensor.
        padding (int): left padding.
    """

    def __init__(self, data, max_dimension, padding=0, scale=1):
        self.data = data
        self.max_dimension = max_dimension
        self.padding = padding
        self._chunks = []
        self.scale = scale

        self.hlen =  0
        self.vlen = 0


    def iter(self):
        flatten = lambda l: [item for sublist in l for item in sublist]
        keys = self.data.keys()
        chunks_dict = {}
        max_num_chunks = 0
        for key in keys:
            if isinstance(self.data[key],torch.Tensor) and len(self.data[key].shape) > 1:
                chunks = max_dimension_split(self.data[key], self.max_dimension, self.padding, dim=2)
                chunks_of_chunks = [max_dimension_split(c, self.max_dimension, self.padding, dim=3) for c in chunks]

                self.vlen = len(chunks)
                self.hlen = len(chunks_of_chunks[0])

                # flatten the list
                chunks_dict[key] = [item for sublist in chunks_of_chunks for item in sublist]
                max_num_chunks = max(max_num_chunks,len(chunks_dict[key]))

        output = {}
        for key in chunks_dict.keys():
            for i in range(len(chunks_dict[key])):
                if isinstance(self.data[key],torch.Tensor):
                    output[key] = chunks_dict[key][i]
                yield output

    def gather(self,tensor):
        self._chunks.append(tensor)

    def clear(self):
        self._chunks = []


    def _concatenate(self,data):
        horiz_chunks = list(chunks_iter(
            data,int(len(data)/self.vlen)))

        vert_chunks = [cat_chunks(h,self.padding*self.scale,3) for h in horiz_chunks]
        return cat_chunks(vert_chunks,self.padding*self.scale,2) # final tensor

    def concatenate(self):
        if isinstance(self._chunks[0],dict):
            ret_data = {}
            for key in self._chunks[0].keys():
                chunks_key = [d[key] for d in self._chunks]
                ret_data[key] = self._concatenate(chunks_key)
            return ret_data
        else:
            return self._concatenate(self._chunks)


