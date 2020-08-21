import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


CPU_PROPERTIES = [10, 9, 8, 7, 6, 6, 6, 6, 6, 6]

LINK_PROPERTIES_BW = [1000, 1000, 500, 400, 100, 100, 100, 100, 100, 100]

LINK_PROPERTIES_LAT = [30, 50, 10, 50, 50, 50, 50, 50, 50, 50]


VNFD_PROPERTIES_SIZE = [0, 4, 3, 3, 2, 2, 2, 1, 1]
VNFD_PROPERTIES_BW = [0, 100, 80, 60, 20, 20, 20, 20, 20]
VNFD_PROPERTIES_LAT = [0, 100, 80, 60, 20, 20, 20, 20, 20]


class Environment(object):
    """
        Attributes:
            num_cpus(int)                           -- Number of hosts
            num_vnfds(int)                          -- Number of VNF descriptors
    """

    def __init__(self, num_cpus, num_vnfds):

        # Environment properties
        self.num_cpus = num_cpus
        self.num_vnfds = num_vnfds
        self.cpu_properties = [{"numVMs": 0} for _ in range(num_cpus)]
        self.link_properties = [{"bandwidth": 0, "latency": 0} for _ in range(num_cpus)]
        self.vnfd_properties = [{"size": 0, "bandwidth": 0, "latency": 0} for _ in range(num_vnfds + 1)]
        self.p_min = 200
        self.p_VM = 100

        # Assign environmental properties
        self._getEnvProperties(num_cpus)
        self._getVnfdProperties(num_vnfds)

        # Environment cell VMs
        self.max_VMs = max([cpu["numVMs"] for cpu in self.cpu_properties])
        self.cells = np.empty((self.num_cpus, self.max_VMs))

        #Initialize Environment variables
        self._initEnv()

    def _initEnv(self):

        # Clear environment
        self.cells[:] = np.nan
        self.cpu_used = np.zeros(self.num_cpus)
        self.link_used = np.zeros(self.num_cpus)

        # Clear placement
        self.service_length = 0
        self.network_service = None
        self.placement = None
        self.first_VMs = None
        self.constraint_occupancy = None
        self.constraint_bandwidth = None
        self.constraint_latency = None
        self.invalid_placement = False
        self.invalid_bandwidth = False
        self.invalid_latency = False

        self.link_latency = 0
        self.cpu_latency = 0


    def _getEnvProperties(self,num_cpus):


            assert num_cpus == len(CPU_PROPERTIES)

            for i in range(num_cpus):

                self.cpu_properties[i]["numVMs"] = CPU_PROPERTIES[i]
                self.link_properties[i]["bandwidth"] = LINK_PROPERTIES_BW[i]
                self.link_properties[i]["latency"] = LINK_PROPERTIES_LAT[i]


    def _getVnfdProperties(self, num_vnfds):

            assert num_vnfds + 1 == len(VNFD_PROPERTIES_SIZE)

            for i in range(num_vnfds + 1):

                self.vnfd_properties[i]["size"] = VNFD_PROPERTIES_SIZE[i]
                self.vnfd_properties[i]["bandwidth"] = VNFD_PROPERTIES_BW[i]
                self.vnfd_properties[i]["latency"] = VNFD_PROPERTIES_LAT[i]


    def _placeVM(self, cpu, vnf):
        """ Place VM """

        occupied_VM = np.nan

        for VM in range(self.cpu_properties[cpu]["numVMs"]):
            if np.isnan(self.cells[cpu][VM]):
                self.cells[cpu][VM] = vnf
                occupied_VM = VM
                break

        return occupied_VM


    def _placeVNF(self, i, cpu, vnf):
        """ Place VNF """

        if self.vnfd_properties[vnf]["size"] <= (self.cpu_properties[cpu]["numVMs"] - self.cpu_used[cpu]):

            for VM in range(self.vnfd_properties[vnf]["size"]):
                occupied_VM = self._placeVM(cpu, vnf)

                # Anotate first VM used by the VNF
                if VM == 0:
                    self.first_VMs[i] = occupied_VM

            self.cpu_used[cpu] += self.vnfd_properties[vnf]["size"]

        else:

            self.cpu_used[cpu] += self.vnfd_properties[vnf]["size"]
            self.first_VMs[i] = -1

    def _computeLink(self):
        """ Compute link usage and link latency """

        self.bandwidth = max([self.vnfd_properties[vnf]["bandwidth"] for vnf in self.network_service])

        for i in range(self.service_length):

            cpu = self.placement[i]

            if i == 0:
                self.link_used[cpu] += self.bandwidth
                self.link_latency += self.link_properties[cpu]["latency"]

            elif cpu != self.placement[i-1]:
                self.link_used[cpu] += self.bandwidth
                self.link_latency += self.link_properties[cpu]["latency"]

            if i == self.service_length - 1:
                self.link_used[cpu] += self.bandwidth
                self.link_latency += self.link_properties[cpu]["latency"]

            elif cpu != self.placement[i+1]:
                self.link_used[cpu] += self.bandwidth
                self.link_latency += self.link_properties[cpu]["latency"]


    def _computeConstraint(self):
        """ Compute reward signals """

        # Check occupancy
        self.constraint_occupancy = 0
        for i in range(self.num_cpus):
            if self.cpu_used[i] > self.cpu_properties[i]["numVMs"]:
                self.invalid_placement = True
                self.constraint_occupancy += self.cpu_used[i] - self.cpu_properties[i]["numVMs"]

        # Check bandwidth
        self.constraint_bandwidth = 0
        for i in range(self.num_cpus):
            if self.link_used[i] > self.link_properties[i]["bandwidth"]:
                self.invalid_bandwidth = True
                self.constraint_bandwidth += self.link_used[i] - self.link_properties[i]["bandwidth"]

        # Check latency
        self.cpu_latency = sum([self.vnfd_properties[vnf]["latency"] for vnf in self.network_service[:self.service_length]])

        self.constraint_latency = 0
        if self.link_latency > self.cpu_latency:
            self.invalid_latency = True
            self.constraint_latency += self.link_latency - self.cpu_latency


    def step(self, length, network_service, placement):
        """ Place network service """

        self.service_length = length
        self.network_service = network_service
        self.placement = placement
        self.first_VMs = -np.ones(length, dtype='int32')

        for i in range(length):
            self._placeVNF(i, placement[i], network_service[i])

        self._computeLink()
        self._computeConstraint()

    def clear(self):

        # Reset environmental variables
        self._initEnv()

    def render(self):
        """ Render environment using MatplotLib """

        # Creates just a figure and only one subplot
        fig, ax = plt.subplots()
        ax.set_title('Environment')

        margin = 3
        margin_ext = 6
        xlim = 100
        ylim = 80

        # Set drawing limits
        plt.xlim(0, xlim)
        plt.ylim(-ylim, 0)

        # Set hight and width for the box
        high = np.floor((ylim - 2 * margin_ext - margin * (self.num_cpus - 1)) / self.num_cpus)
        wide = np.floor((xlim - 2 * margin_ext - margin * (self.max_VMs - 1)) / self.max_VMs)

        plt.text(10, 1, "Cstr occ: {}".format(self.constraint_occupancy), ha="center", family='sans-serif', size=8)
        plt.text(20, 1, "Cstr bw: {}".format(self.constraint_bandwidth), ha="center", family='sans-serif', size=8)
        plt.text(30, 1, "Cstr lat: {}".format(self.constraint_latency), ha="center", family='sans-serif', size=8)


        # Plot VM labels
        for VM in range(self.max_VMs):
            x = wide * VM + VM * margin + margin_ext
            plt.text(x + 0.5 * wide, -3, "VM{}".format(VM), ha="center", family='sans-serif', size=8)

        # Plot cpu labels & placement empty boxes
        for cpu in range(self.num_cpus):
            y = -high * (cpu + 1) - (cpu) * margin - margin_ext
            plt.text(0, y + 0.5 * high, "cpu{}".format(cpu), ha="center", family='sans-serif', size=8)

            for VM in range(self.cpu_properties[cpu]["numVMs"]):
                x = wide * VM + VM * margin + margin_ext
                rectangle = mpatches.Rectangle((x, y), wide, high, linewidth=1, edgecolor='black', facecolor='none')
                ax.add_patch(rectangle)

        # Select service_length colors from a colormap
        cmap = plt.cm.get_cmap('hot')
        colormap = [cmap(np.float32(i+1)/(self.service_length+1)) for i in range(self.service_length)]

        # Plot service boxes
        for idx in range(self.service_length):
            vnf = self.network_service[idx]
            cpu = self.placement[idx]
            first_VM = self.first_VMs[idx]

            for k in range(self.vnfd_properties[vnf]["size"]):

                # Plot ONLY if it is a valid placement
                if first_VM != -1:
                    VM = first_VM + k
                    x = wide * VM + VM * margin + margin_ext
                    y = -high * (cpu + 1) - cpu * margin - margin_ext
                    rectangle = mpatches.Rectangle((x, y), wide, high, linewidth=0, facecolor=colormap[idx], alpha=.9)
                    ax.add_patch(rectangle)
                    plt.text(x + 0.5 * wide, y + 0.5 * high, "vnf{}".format(vnf), ha="center", family='sans-serif', size=8)

        plt.axis('off')
        plt.show()


if __name__ == "__main__":

    # Define environment
    num_cpus = 10
    num_vnfds = 8

    env = Environment(num_cpus, num_vnfds)

    # Allocate service in the environment
    service_length = 8
    network_service = [ 4, 8, 1, 4, 3, 6, 6, 8]
    placement = [3, 3, 2, 1, 1, 0, 0, 0]


    env.step(service_length, network_service, placement)

    print("Placement Invalid: ", env.invalid_placement)
    print("Link used: ", env.link_used, "Invalid: ", env.invalid_bandwidth)
    print("CPU Latency: ", env.cpu_latency, "Link Latency: ", env.link_latency, "Invalid: ", env.invalid_latency)
    print("Constraint_occupancy: ", env.constraint_occupancy)
    print("Constraint_bandwidth: ", env.constraint_bandwidth)
    print("Constraint_latency: ", env.constraint_latency)

    print(env.cpu_properties)
    print(env.link_properties)
    print(env.vnfd_properties)
    print(env.bandwidth)

    env.render()
    env.clear()



