from simpful import *
import matplotlib.pyplot as plt
from numpy import linspace, array


# Stwórz obiekt FIS (Fuzzy Inference System)
def define_FS():
    FS = FuzzySystem(show_banner=False)

    # Zdefiniuj wartości i zmienne lingwistyczne dla wejść do systemu
    o1 = FuzzySet(function=Triangular_MF(a=0, b=0, c=5), term="kiepska")
    o2 = FuzzySet(function=Triangular_MF(a=0, b=5, c=10), term="srednia")
    o3 = FuzzySet(function=Triangular_MF(a=5, b=10, c=10), term="wspaniala")
    FS.add_linguistic_variable("Obsluga", LinguisticVariable([o1, o2, o3], concept="Jakość obsługi",
                                                             universe_of_discourse=[0, 10]))

    j1 = FuzzySet(function=Triangular_MF(a=0, b=0, c=10), term="niesmaczne")
    j2 = FuzzySet(function=Triangular_MF(a=0, b=10, c=10), term="pyszne")
    FS.add_linguistic_variable("Jedzenie", LinguisticVariable([j1, j2], concept="Jakość jedzenia",
                                                              universe_of_discourse=[0, 10]))

    # Zdefiniuj wartości i zmienne lingwistyczne dla wyjść systemu
    n1 = FuzzySet(function=Triangular_MF(a=0, b=0, c=10), term="niski")
    n2 = FuzzySet(function=Triangular_MF(a=0, b=10, c=20), term="sredni")
    n3 = FuzzySet(function=Trapezoidal_MF(a=10, b=20, c=25, d=25), term="chojny")
    FS.add_linguistic_variable("Napiwek", LinguisticVariable([n1, n2, n3],
                                                             universe_of_discourse=[0, 25]))

    # Zdefiniuj reguły systemu
    r1 = "IF (Obsluga IS kiepska) OR (Jedzenie IS niesmaczne) THEN (Napiwek IS niski)"
    r2 = "IF (Obsluga IS srednia) THEN (Napiwek IS sredni)"
    r3 = "IF (Obsluga IS wspaniala) OR (Jedzenie IS pyszne) THEN (Napiwek IS chojny)"
    FS.add_rules([r1, r2, r3])

    return FS


def inference_FS(fis, service_value, food_value):
    # ustaw wartości wejść
    fis.set_variable("Obsluga", service_value)
    fis.set_variable("Jedzenie", food_value)
    print(fis.Mamdani_inference(["Napiwek"]))
    print(fis.get_firing_strengths({"Obsluga": [service_value], "Jedzenie": [food_value]}))


def plot_decision_surface(fis):
    # Plotting surface
    xs = []
    ys = []
    zs = []
    DIVs = 20
    for x in linspace(0, 10, DIVs):
        for y in linspace(0, 10, DIVs):
            fis.set_variable("Jedzenie", x)
            fis.set_variable("Obsluga", y)
            tip = fis.inference()['Napiwek']
            xs.append(x)
            ys.append(y)
            zs.append(tip)
    xs = array(xs)
    ys = array(ys)
    zs = array(zs)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.plot_trisurf(xs, ys, zs, vmin=0, vmax=25, cmap='gnuplot2')
    ax.set_xlabel("Jedzenie")
    ax.set_ylabel("Obsluga")
    ax.set_zlabel("Napiwek")
    ax.set_title("Simpful", pad=20)
    ax.set_zlim(0, 25)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    fis = define_FS()
    inference_FS(fis, -1, 0)
    inference_FS(fis, 0, 0)
    inference_FS(fis, 4, 8)
    inference_FS(fis, 4, 2)
    inference_FS(fis, 4, 9)
    inference_FS(fis, 8, 9)
    inference_FS(fis, 10, 10)
    fis.produce_figure()
    fis.plot_variable("Obsluga", highlight="wspaniala")
    fis.plot_variable("Jedzenie", TGT=7)
    plot_decision_surface(fis)
