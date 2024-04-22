import cfusdlog
import matplotlib.pyplot as plt


def decode_data(path):
    data_usd = cfusdlog.decode(path)
    data = data_usd['fixedFrequency']
    return data

data = decode_data('../flight_data/flight00')
x = [i for i in data["stateEstimate.x"]]
y = [i for i in data["stateEstimate.y"]]
z = [i for i in data["stateEstimate.z"]]
ax = plt.axes(projection="3d")
ax.plot3D(x, y, z)
plt.show()
