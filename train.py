import caffe
from config import configData


def initCaffe():
    caffe.set_device(0)
    caffe.set_mode_gpu()

def main():
    initCaffe()

    solver = caffe.get_solver(configData["solverProtoFile"])
    solver.step(solver.param.max_iter)

if __name__ == "__main__":
    main()
