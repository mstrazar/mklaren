#encoding="utf-8"
from gzip import open as gzopen
from numpy import loadtxt, zeros, array
from os import environ
from os.path import join, dirname, realpath

DELVE_PATH = join(dirname(realpath(__file__)), "delve")

class Specdict(dict):
    """
    Default dict for specification of datasets.

    """
    TARGET = "TARGET"

    def __init__(self):
        """
        num_vars is the current number of variables in the mapping.

        """
        super(Specdict, self).__init__()
        self.num_vars = 0
        self.labels = dict()


    def add_variable(self, inx, name, values=None):
        """
        Add a variable to dictionary.
        Categorical variables are indexed by original index (inx)
            and value (val).
        Continuous variables are index by original index (inx).
        """

        # Categorical
        if values is not None:
            assert values is not None
            for val in values:
                self[inx, val] = self.num_vars
                self.labels[self.num_vars] = name + "_" + str(val)
                self.num_vars += 1

        # Continuous
        else:
            self[inx] = self.num_vars
            self.labels[self.num_vars] = name
            self.num_vars += 1


    def add_target(self, inx):
        """
        Add target variable y.
        """
        self[inx] = self.TARGET


def parse_spec(specfile):
    """
    Parse specification file as a dict.

    :param specfile
        Path to the .spec file.
    :return
        Mapping original index -> matrix index.
    """
    sd = Specdict()
    fp = open(specfile, "r")
    inx = 0                  # Original index
    line = fp.readline()
    while line:
        if not line.startswith("#"):
            ls = line.strip().split()
            name = ls[1]
            typ  = ls[2]  # If type is categorical,
                          # expect range to be stored in the neighbouring column
                          # All categorical variables are strings.

            if typ == "c":
                rng = ls[3]
                if ".." in rng:
                    start, end = map(int, rng.split(".."))
                    values = range(start, end+1)
                    values = map(str, values)
                elif "," in rng:
                    values = rng.split(",")
                sd.add_variable(inx=inx, name=name,
                                values=values)
            elif typ == "u":
                sd.add_variable(inx=inx, name=name,)

            elif typ == "y":
                sd.add_target(inx)

            else:
                pass


            inx += 1
        line = fp.readline()

    return sd



def load_delve(dataset_path, dataset_spec, n=None):
    """
        Load an delve dataset. Specification is given by the spec file.

        :param dataset_path
            Path to the .data.gz file.
        :param dataset_spec
            Path to the .spec file.
        :param n
            If defined, read only first n rows.

        :return
            Dictionary data, target.
    """
    rdict = dict()
    sd    = parse_spec(dataset_spec)
    fp    = gzopen(dataset_path, "r")
    line  = str(fp.readline())
    count = 0

    X = list()
    y = list()

    while line:

        if line.count('\\'):
            # Must read another line
            line = line.strip().replace("\\", "") + str(fp.readline())

        x = zeros((sd.num_vars, ))
        for i, v in enumerate(line.strip().split()):
            if i in sd:
                if sd[i] == sd.TARGET:
                    y.append(float(v))
                else:
                    j = sd[i]
                    x[j] = float(v)
            elif (i, v) in sd:
                j = sd[i, v]
                x[j] = 1
            else:
                pass

        X.append(x)

        line = str(fp.readline())
        count += 1

        if n is not None and count == n:
            break

    rdict["data"]   = array(X)
    rdict["target"] = array(y)
    rdict["labels"] = [sd.labels[i] for i in range(len(X[0]))]

    return rdict





def load_abalone(n=None):
    return load_delve(dataset_path=join(DELVE_PATH, "abalone", "Dataset.data.gz"),
                      dataset_spec=join(DELVE_PATH, "abalone", "Dataset.spec"),
                      n = n)

def load_boston(n=None):
    return load_delve(dataset_path=join(DELVE_PATH, "boston", "Dataset.data.gz"),
                      dataset_spec=join(DELVE_PATH, "boston", "Dataset.spec"),
                      n = n)

def load_census_house(n=None):
    return load_delve(dataset_path=join(DELVE_PATH, "census-house", "Dataset.data.gz"),
                      dataset_spec=join(DELVE_PATH, "census-house", "Dataset.spec"),
                      n = n)

def load_comp_activ(n=None):
    return load_delve(dataset_path=join(DELVE_PATH, "comp-activ", "Dataset.data.gz"),
                      dataset_spec=join(DELVE_PATH, "comp-activ", "Dataset.spec"),
                      n = n)




def load_bank(typ="8fh", n=None):
    return load_delve(dataset_path=join(DELVE_PATH, "bank", "bank-%s" % typ, "Dataset.data.gz"),
                      dataset_spec=join(DELVE_PATH, "bank", "bank-%s" % typ, "Dataset.spec"),
                      n = n)

def load_pumadyn(typ="8fh", n=None):
    return load_delve(dataset_path=join(DELVE_PATH, "pumadyn", "pumadyn-%s" % typ, "Dataset.data.gz"),
                      dataset_spec=join(DELVE_PATH, "pumadyn", "pumadyn-%s" % typ, "Dataset.spec"),
                      n = n)

def load_kin(typ="8fh", n=None):
    return load_delve(dataset_path=join(DELVE_PATH, "kin", "kin-%s" % typ, "Dataset.data.gz"),
                      dataset_spec=join(DELVE_PATH, "kin", "kin-%s" % typ, "Dataset.spec"),
                      n = n)




if __name__ == "__main__":

    for load in [
        load_abalone,
        load_census_house,
        load_comp_activ,
        load_boston,
    ]:

        data = load()
        X = data["data"]
        y = data["target"]
        labels = data["labels"]
        print(X.shape, y.shape)
        print(labels)
        assert X.sum() != 0
        assert y.sum() != 0
        print


    for load in [
        load_bank,
        load_pumadyn,
        load_kin,
    ]:
        for num in "8", "32":
            for t in ["fh", "fm", "nh", "nm"]:
                typ = "%s%s" % (num, t)
                data = load(typ=typ)
                X = data["data"]
                y = data["target"]
                labels = data["labels"]
                print(load, X.shape, y.shape)
                assert X.sum() != 0
                assert y.sum() != 0
                print()



