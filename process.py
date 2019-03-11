import numpy as np
import xarray as xr
import re

def distance(val, ref):
    return abs(ref - val)
vectDistance = np.vectorize(distance)

def getClosest(sortedMatrix, column, val):
    while len(sortedMatrix) > 3:
        half = int(len(sortedMatrix) / 2)
        sortedMatrix = sortedMatrix[-half - 1:] if sortedMatrix[half, column] < val else sortedMatrix[: half + 1]
    if len(sortedMatrix) == 1:
        result = sortedMatrix[0].copy()
        result[column] = val
        return result
    else:
        safecopy = sortedMatrix.copy()
        safecopy[:, column] = vectDistance(safecopy[:, column], val)
        minidx = np.argmin(safecopy[:, column])
        safecopy = safecopy[minidx, :].A1
        safecopy[column] = val
        return safecopy

def convert(column, samples, matrix):
    return np.matrix([getClosest(matrix, column, t) for t in samples])

def valueOrEmptySet(k, d):
    return (d[k] if isinstance(d[k], set) else {d[k]}) if k in d else set()

def mergeDicts(d1, d2):
    """
    Creates a new dictionary whose keys are the union of the keys of two
    dictionaries, and whose values are the union of values.

    Parameters
    ----------
    d1: dict
        dictionary whose values are sets
    d2: dict
        dictionary whose values are sets

    Returns
    -------
    dict
        A dict whose keys are the union of the keys of two dictionaries,
    and whose values are the union of values

    """
    res = {}
    for k in d1.keys() | d2.keys():
        res[k] = valueOrEmptySet(k, d1) | valueOrEmptySet(k, d2)
    return res

def extractCoordinates(filename):
    """
    Scans the header of an Alchemist file in search of the variables.

    Parameters
    ----------
    filename : str
        path to the target file
    mergewith : dict
        a dictionary whose dimensions will be merged with the returned one

    Returns
    -------
    dict
        A dictionary whose keys are strings (coordinate name) and values are
        lists (set of variable values)

    """
    with open(filename, 'r') as file:
        regex = re.compile(' (?P<varName>[a-zA-Z]+) = (?P<varValue>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?),?')
        dataBegin = re.compile('\d')
        for line in file:
            match = regex.findall(line)
            if match:
                return {var : float(value) for var, value in match}
            elif dataBegin.match(line[0]):
                return {}

def extractVariableNames(filename):
    """
    Gets the variable names from the Alchemist data files header.

    Parameters
    ----------
    filename : str
        path to the target file

    Returns
    -------
    list of list
        A matrix with the values of the csv file

    """
    with open(filename, 'r') as file:
        dataBegin = re.compile('\d')
        lastHeaderLine = ''
        for line in file:
            if dataBegin.match(line[0]):
                break
            else:
                lastHeaderLine = line
        if lastHeaderLine:
            regex = re.compile(' (?P<varName>\S+)')
            return regex.findall(lastHeaderLine)
        return []

def openCsv(path):
    """
    Converts an Alchemist export file into a list of lists representing the matrix of values.

    Parameters
    ----------
    path : str
        path to the target file

    Returns
    -------
    list of list
        A matrix with the values of the csv file

    """
    regex = re.compile('\d')
    with open(path, 'r') as file:
        lines = filter(lambda x: regex.match(x[0]), file.readlines())
        return [[float(x) for x in line.split()] for line in lines]

if __name__ == '__main__':
    # CONFIGURE SCRIPT
    directory = 'data'
    pickleOutput = 'data_summary'
    experiments = ['cloud', 'edge']
    floatPrecision = '{: 0.2f}'
    seedVars = ['seed']
    minTime = 65
    maxTime = 665
    timeSamples = int(maxTime - minTime) // 2
    timeColumnName = 'time'
    logarithmicTime = False
    allowPreprocess = True
    
    # Setup libraries
    np.set_printoptions(formatter={'float': floatPrecision.format})
    # Read the last time the data was processed, reprocess only if new data exists, otherwise just load
    import pickle
    import os
    newestFileTime = max(os.path.getmtime(directory + '/' + file) for file in os.listdir(directory))
    try:
        lastTimeProcessed = pickle.load(open('timeprocessed', 'rb'))
    except:
        lastTimeProcessed = -1
    shouldRecompute = allowPreprocess and newestFileTime != lastTimeProcessed
    if not shouldRecompute:
        try:
            means = pickle.load(open(pickleOutput + '_mean', 'rb'))
            stdevs = pickle.load(open(pickleOutput + '_std', 'rb'))
        except:
            shouldRecompute = True
    if shouldRecompute:
        timefun = np.logspace if logarithmicTime else np.linspace
        means = {}
        stdevs = {}
        for experiment in experiments:
            # Collect all files for the experiment of interest
            import fnmatch
            allfiles = filter(lambda file: fnmatch.fnmatch(file, experiment + '_*.txt'), os.listdir(directory))
            allfiles = [directory + '/' + name for name in allfiles]
            allfiles.sort()
            # From the file name, extract the independent variables
            dimensions = {}
            for file in allfiles:
                dimensions = mergeDicts(dimensions, extractCoordinates(file))
            dimensions = {k: sorted(v) for k, v in dimensions.items()}
            # Add time to the independent variables
            dimensions[timeColumnName] = range(0, timeSamples)
            # Compute the matrix shape
            shape = tuple(len(v) for k, v in dimensions.items())
            # Prepare the Dataset
            dataset = xr.Dataset()
            for k, v in dimensions.items():
                dataset.coords[k] = v
            varNames = extractVariableNames(allfiles[0])
            for v in varNames:
                if v != timeColumnName:
                    novals = np.ndarray(shape)
                    novals.fill(float('nan'))
                    dataset[v] = (dimensions.keys(), novals)
            # Compute maximum and minimum time, create the resample
            timeColumn = varNames.index(timeColumnName)
            allData = { file: np.matrix(openCsv(file)) for file in allfiles }
            computeMin = minTime is None
            computeMax = maxTime is None
            if computeMax:
                maxTime = float('-inf')
                for data in allData.values():
                    maxTime = max(maxTime, data[-1, timeColumn])
            if computeMin:
                minTime = float('inf')
                for data in allData.values():
                    minTime = min(minTime, data[0, timeColumn])
            timeline = timefun(minTime, maxTime, timeSamples)
            # Resample
            for file in allData:
                print("Working on " + file)
                allData[file] = convert(timeColumn, timeline, allData[file])
            # Populate the dataset
            for file, data in allData.items():
                dataset[timeColumnName] = timeline
                for idx, v in enumerate(varNames):
                    if v != timeColumnName:
                        darray = dataset[v]
                        experimentVars = extractCoordinates(file)
                        darray.loc[experimentVars] = data[:, idx].A1
            # Fold the dataset along the seed variables, producing the mean and stdev datasets
            means[experiment] = dataset.mean(seedVars)
            stdevs[experiment] = dataset.std(seedVars)
        # Save the datasets
        pickle.dump(means, open(pickleOutput + '_mean', 'wb'), protocol=-1)
        pickle.dump(stdevs, open(pickleOutput + '_std', 'wb'), protocol=-1)
        pickle.dump(newestFileTime, open('timeprocessed', 'wb'))

    # Prepare the charting system
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.cm as cmx
    import matplotlib.patches as patches
    figure_size=(6, 3.5)
    matplotlib.rcParams.update({'axes.titlesize': 13})
    matplotlib.rcParams.update({'axes.labelsize': 12})
#    colormap = cmx.gist_rainbow
    colormap = lambda x: (lambda alpha: cmx.viridis(alpha * x))(0.9)
    
    # Edge delay and size with range and ap-count
    packet_size = 'packet_size[Mean]'
    delay = 'mean_delay[Mean]'
    crowd = 'crowd[Sum]'
    energy = '<value>@neighbor_count[Sum]'
    titlebase = {
        packet_size: 'Packet size',
        delay: 'Delay',
        crowd: 'Users notified',
        energy: 'Appr. energy use'
    }
    ylabels = {
        packet_size: 'Packet size (Bytes)',
        delay: 'Mean packet delay (s)',
        energy: 'Mean data processed (B/s)',
        crowd: 'Notified users'
    }
    x = means['edge']['baserange'] / 2
    for var in [delay, crowd, packet_size, energy]:
        merge_data = means['edge'].mean('time').mean('bandwidth')
#        merge_data = means['edge'].mean('time').sel(bandwidth = means['edge']['bandwidth'][7])
        fig = plt.figure(figsize = figure_size)
        ax = fig.add_subplot(1, 1, 1)
        ax.set_xlabel("Wireless range (m)")
        ax.set_ylabel(ylabels[var])
        ax.set_xscale('log')
        ax.set_title(titlebase[var] + ' with range and edge server count')
        index = 0
        for ap_count in merge_data['accessPointCount']:
            y = merge_data.data_vars[var].sel({'accessPointCount': ap_count})
            if (var == energy):
                y *= merge_data.data_vars[packet_size].sel({'accessPointCount': ap_count})
                ax.set_yscale('log')
                ax.set_ylim(top = 1.8e8)
            ax.plot(x, y, label=str(int(ap_count)), color=colormap(index / 9))
            index = index + 1
        ax.set_xlim(min(x), max(x))
        ylim = {
            delay: (0, 2.7),
            crowd: (0, 450),
            packet_size: (0, 1300),
        }.get(var, ax.get_ylim())
        ypos = {
            packet_size: 1100,
            delay: 2.3,
            energy: 4e2,
            crowd: 425
        }[var]
        ax.set_ylim(ylim)
        ax.add_patch(patches.Rectangle((46, ax.get_ylim()[0]), 92 - 46, ax.get_ylim()[1]-ax.get_ylim()[0], alpha=0.15, color='black'))
        ax.text(x=36, y=ypos, color='black', s='typical Wi-Fi range')
        ax.legend(ncol=5)
        if (var == crowd):
            ax.legend(ncol=1)
#            ax.set_ylim(bottom=-100, top=ax.get_ylim()[1])
        fig.tight_layout()
        fig.savefig('edge-range'+''.join(filter(str.isalpha, titlebase[var]))+'.pdf', transparent=True)
    
    # Edge delay and size with bandwidth and ap-count
    x = means['edge']['bandwidth']
    for var in [delay, crowd, packet_size, energy]:
        merge_data = means['edge'].mean('time').mean('baserange')
        fig = plt.figure(figsize = figure_size)
        ax = fig.add_subplot(1, 1, 1)
        ax.set_xlabel("Data rate (B/s)")
        ax.set_ylabel(ylabels[var])
        ax.set_xscale('log')
        ax.set_title(titlebase[var] + ' with data rate and edge server count')
        index = 0
        for ap_count in merge_data['accessPointCount']:
            y = merge_data.data_vars[var].sel({'accessPointCount': ap_count})
            if (var == energy):
                y *= merge_data.data_vars[packet_size].sel({'accessPointCount': ap_count})
                ax.set_yscale('log')
            ax.plot(x, y, label=str(int(ap_count)), color=colormap(index / 9))
            index = index + 1
        ax.set_xlim(min(x), max(x))
        ncol = {
            packet_size: 5,
            delay: 5,
            energy: 5,
            crowd: 5
        }[var]
        ylim = {
            packet_size: (380, ax.get_ylim()[1]),
            crowd: (0, 365),
            energy: (2e3, ax.get_ylim()[1]),
        }.get(var, ax.get_ylim())
        ax.set_ylim(ylim)
        ax.legend(ncol=ncol)
        ax.add_patch(patches.Rectangle((5.5e6 / 8, 0), 100e6/8 - 5.5e6 / 8, ax.get_ylim()[1], alpha=0.15, color='black'))
        texty = {
            crowd: 30,
            packet_size: 580,
            energy: 2e4,
        }.get(var, ax.get_ylim()[1] / 2)
        ax.text(x=5e5, y=texty, color='black', s='typical Wi-Fi data rate')
        fig.tight_layout()
        fig.savefig('edge-datarate'+''.join(filter(str.isalpha, titlebase[var]))+'.pdf')
    
    # Cloud delay and size with bandwidth and range
    energy = 'neighbor_count[Sum]'
    titlebase[energy] = 'Appr. energy use'
    ylabels[energy] = 'Mean data processed (B/s)'
    x = means['cloud']['bandwidth']
    mobile_technologies = {
        "EDGE": [0.1e6 / 8, 0.3e6 / 8, 'black'],
        "HSPA": [1.5e6 / 8, 7.2e6 / 8, 'red'],
        "HSPA+": [4e6 / 8, 21e6 / 8, 'green'],
        "LTE-AC6": [24e6 / 8, 300e6 / 8, 'blue']
    }
    for var in [delay, crowd, packet_size, energy]:
        merge_data = means['cloud'].mean('time').mean('propagationDelay')
        fig = plt.figure(figsize = figure_size)
        ax = fig.add_subplot(1, 1, 1)
        ax.set_xlabel("Data rate (B/s)")
        ax.set_ylabel(ylabels[var])
        ax.set_xscale('log')
        ax.set_title(titlebase[var] + ' with data rate and computation effort')
        index = 0
        for _range in merge_data['baserange']:
            y = merge_data.data_vars[var].sel({'baserange': _range})
            if (var == energy):
                y *= merge_data.data_vars[packet_size].sel({'baserange': _range})
                ax.set_yscale('log')
            ax.plot(x, y, label=str(int(_range)), color=colormap(index / 9))
            index = index + 1
        ylim = {
            delay: (0, 3.6),
            energy: (3e4, 4e8),
        }.get(var, ax.get_ylim())
        ax.set_ylim(ylim)
        texty = {
            delay: 3.1,
            crowd: 180,
            energy: 2e6,
        }.get(var, ax.get_ylim()[1] / 2)
        for mobile_name, mobile_perf in mobile_technologies.items():
            bw_min = mobile_perf[0] / 2
            bw_max = mobile_perf[1] / 2
            ax.add_patch(patches.Rectangle((bw_min, 0), bw_max-bw_min, ax.get_ylim()[1], alpha=0.15, color=mobile_perf[2]))
            ax.text(x=(bw_max - bw_min)*0.05 + bw_min, y=texty, color='black', s=mobile_name)
        ax.legend(ncol=5)
        ax.set_xlim(min(x), max(x))
        fig.tight_layout()
        fig.savefig('cloud-bandwidth'+''.join(filter(str.isalpha, titlebase[var]))+'.pdf')
    
    # Cloud delay and size with propagation delay and range
    x = means['cloud']['propagationDelay']
    for var in [delay, crowd, packet_size, energy]:
        merge_data = means['cloud'].mean('time').mean('bandwidth')
        fig = plt.figure(figsize = figure_size)
        ax = fig.add_subplot(1, 1, 1)
        ax.set_xlabel("Propagation delay (s)")
        ax.set_ylabel(ylabels[var])
        ax.set_xscale('log')
        ax.set_title(titlebase[var] + ' with prop. delay and comp. effort')
        index = 0
        for _range in merge_data['baserange']:
            y = merge_data.data_vars[var].sel({'baserange': _range})
            if (var == energy):
                y *= merge_data.data_vars[packet_size].sel({'baserange': _range})
                ax.set_yscale('log')
            ax.plot(x, y, label=str(int(_range)), color=colormap(index / 9))
            index = index + 1
        ylim = {
            energy: (3e4, 5e8)
        }.get(var, ax.get_ylim())
        ax.set_ylim(ylim)
        texty = {
            energy: 5e5
        }.get(var, (ylim[1] - ylim[0]) * 0.2 + ylim[0])
        ax.add_patch(patches.Rectangle((50e-3, 0), 450e-3, ax.get_ylim()[1], alpha=0.15, color='black'))
        ax.set_xlim(min(x), max(x))
        ax.text(x=7e-2, y=texty, color='black', s='typical delay')
        ax.legend(ncol=5)
        fig.tight_layout()
        fig.savefig('cloud-propagation'+''.join(filter(str.isalpha, titlebase[var])) + '.pdf')
    
    # Comparisons
    def comparisonChart(ycol, fun = lambda table, ycol: table.data_vars[ycol]):
        divergingmixcolormap = lambda x: cmx.winter(1 - x * 2) if x < 0.5 else cmx.YlOrRd((x - 0.5) * 2 * 0.6 + 0.3)
        edge_sparse = means['edge'].sel({
                'baserange': means['edge']['baserange'][5],
                'bandwidth': means['edge']['bandwidth'][6],
                'accessPointCount': means['edge']['accessPointCount'][5]})
        edge_dense = means['edge'].sel({
                'baserange': means['edge']['baserange'][5],
                'bandwidth': means['edge']['bandwidth'][6],
                'accessPointCount': means['edge']['accessPointCount'][7]})
        edge_denser = means['edge'].sel({
                'baserange': means['edge']['baserange'][5],
                'bandwidth': means['edge']['bandwidth'][6],
                'accessPointCount': means['edge']['accessPointCount'][8]})
        cloud_2g = means['cloud'].sel({
                'baserange': means['cloud']['baserange'][2],
                'bandwidth': means['cloud']['bandwidth'][1],
                'propagationDelay': means['cloud']['propagationDelay'][7]})
        cloud_3g = means['cloud'].sel({
                'baserange': means['cloud']['baserange'][3],
                'bandwidth': means['cloud']['bandwidth'][4],
                'propagationDelay': means['cloud']['propagationDelay'][6]})
        cloud_4g = means['cloud'].sel({
                'baserange': means['cloud']['baserange'][4],
                'bandwidth': means['cloud']['bandwidth'][7],
                'propagationDelay': means['cloud']['propagationDelay'][5]})
        fig = plt.figure(figsize = figure_size)
        ax = fig.add_subplot(1, 1, 1)
        ax.set_xlabel("Simulated time (s)")
    #    ax.set_yscale('log')
        x = edge_sparse['time'] - minTime
        ax.plot(x, fun(edge_sparse, ycol), label='edge-sparse', color=divergingmixcolormap(0.5/5))
        ax.plot(x, fun(edge_dense, ycol), label='edge-dense', color=divergingmixcolormap(1.5/5))
        ax.plot(x, fun(edge_denser, ycol), label='edge-denser', color=divergingmixcolormap(2.4/5))
        ax.plot(x, fun(cloud_2g, ycol), label='cloud-2G', color=divergingmixcolormap(3/5))#, linestyle=':')
        ax.plot(x, fun(cloud_3g, ycol), label='cloud-3G', color=divergingmixcolormap(4/5))#, linestyle='--')
        ax.plot(x, fun(cloud_4g, ycol), label='cloud-LTE', color=divergingmixcolormap(5/5))#, linestyle='-.')
        ax.set_xlim(0, maxTime - minTime)
        return (fig, ax)
    
    # Comparison
    delay = 'mean_delay[Mean]'
    fig, ax = comparisonChart(delay)
    ax.set_ylabel(ylabels[delay])
    ax.set_title(titlebase[delay] + ': edge/cloud comparison')
    ax.set_yscale('log')
    ax.legend(ncol=2)#, bbox_to_anchor=(0.37, 0.58))
    fig.tight_layout()
    fig.savefig('edge-cloud-delay.pdf')
    
    # Quality
    fig, ax = comparisonChart('crowd[Sum]')
    ax.set_ylabel(ylabels[crowd])
    ax.set_title(titlebase[crowd] + ': edge/cloud comparison')
    ax.set_ylim(180, 520)
    ax.legend(ncol=2, bbox_to_anchor=(0.4, 0.5))
    fig.tight_layout()
    fig.savefig('edge-cloud-quality.pdf')
    
    # Energy
    energy_edge = '<value>@neighbor_count[Sum]'
    def computeEnergy(table, ycol):
        import numpy as np
        isCloud = np.isnan(np.nanmean(table[energy_edge]))
        pick = lambda col: table.data_vars[col]
        return pick(ycol) * pick(energy) if isCloud else pick(energy_edge)
    fig, ax = comparisonChart(packet_size, computeEnergy)
    ax.set_ylabel(ylabels[energy])
    ax.set_yscale('log')
    ax.set_ylim((3e2, 3e7))
    ax.set_title(titlebase[energy] + ': edge/cloud comparison')
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig('edge-cloud-energy.pdf')
    
    #means['edge'].sel(bandwidth=1e7).mean('time').data_vars['packet_size[Mean]']
    #cut pdf borders
    try:
        import subprocess
        for image in filter(lambda file: fnmatch.fnmatch(file, '*.pdf'), os.listdir('.')):
            subprocess.call(["krop", "--autotrim", "--go", '-o', image, image])
    except OSError as e:
        print('Unable to trim PDFS using krop: ' + e)
