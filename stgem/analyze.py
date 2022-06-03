from stgem.test_repository import TestRepository


def test_repository_contour_plot(tr: TestRepository, show="sut_output"):
    I, Osut , Oobjectives = tr.get()
    if show=="sut_output":
        O= Osut
    else:
        O= Oobjectives

    """
    TODO: plot a matplot countourplot of the variables I,O
    see https://matplotlib.org/stable/gallery/images_contours_and_fields/contour_demo.html
    """
