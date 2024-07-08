import numpy as np
from numpy import array as array
from lasr_vision_feature_extraction.categories_and_attributes import (
    CategoriesAndAttributes,
)
import cv2
import matplotlib.pyplot as plt


def _softmax(x: list[float]) -> list[float]:
    """Compute softmax values for each set of scores in x without using NumPy."""
    # First, compute e^x for each value in x
    exp_values = [_exp(val) for val in x]
    # Compute the sum of all e^x values
    sum_of_exp_values = sum(exp_values)
    # Now compute the softmax value for each original value in x
    softmax_values = [exp_val / sum_of_exp_values for exp_val in exp_values]
    return softmax_values


def _exp(x):
    """Compute e^x for a given x. A simple implementation of the exponential function."""
    return 2.718281828459045**x  # Using an approximation of Euler's number e


class ImageWithMasksAndAttributes:
    def __init__(
        self,
        image: np.ndarray,
        masks: dict[str, np.ndarray],
        attributes: dict[str, float],
        categories_and_attributes: CategoriesAndAttributes,
    ):
        self.image: np.ndarray = image
        self.masks: dict[str, np.ndarray] = masks
        self.attributes: dict[str, float] = attributes
        self.categories_and_attributes: CategoriesAndAttributes = (
            categories_and_attributes
        )

        self.plane_attribute_dict: dict[str, float] = {}
        for attribute in self.categories_and_attributes.plane_attributes:
            self.plane_attribute_dict[attribute] = self.attributes[attribute]

        self.selective_attribute_dict: dict[str, dict[str, float]] = {}
        for category in sorted(
            list(self.categories_and_attributes.selective_attributes.keys())
        ):
            self.selective_attribute_dict[category] = {}
            temp_list: list[float] = []
            for attribute in self.categories_and_attributes.selective_attributes[
                category
            ]:
                temp_list.append(self.attributes[attribute])
            softmax_list = _softmax(temp_list)
            for i, attribute in enumerate(
                self.categories_and_attributes.selective_attributes[category]
            ):
                self.selective_attribute_dict[category][attribute] = softmax_list[i]

    def describe(self) -> str:
        # abstract method
        pass


def _max_value_tuple(some_dict: dict[str, float]) -> tuple[str, float]:
    max_key = max(some_dict, key=some_dict.get)
    return max_key, some_dict[max_key]


class ImageOfPerson(ImageWithMasksAndAttributes):
    def __init__(
        self,
        image: np.ndarray,
        masks: dict[str, np.ndarray],
        attributes: dict[str, float],
        categories_and_attributes: CategoriesAndAttributes,
    ):
        super().__init__(image, masks, attributes, categories_and_attributes)

    @classmethod
    def from_parent_instance(
        cls, parent_instance: ImageWithMasksAndAttributes
    ) -> "ImageOfPerson":
        """
        Creates an instance of ImageOfPerson using the properties of an
        instance of ImageWithMasksAndAttributes.
        """
        return cls(
            image=parent_instance.image,
            masks=parent_instance.masks,
            attributes=parent_instance.attributes,
            categories_and_attributes=parent_instance.categories_and_attributes,
        )

    def describe(self) -> dict:
        has_hair = self.attributes["hair"] - 0.5
        hair_colour = _max_value_tuple(self.selective_attribute_dict["hair_colour"])[0]
        hair_shape = _max_value_tuple(self.selective_attribute_dict["hair_shape"])[0]
        facial_hair = 1 - self.attributes["No_Beard"] - 0.5
        glasses = self.attributes["Eyeglasses"] - 0.5
        hat = self.attributes["Wearing_Hat"] - 0.5

        if hat >= 0.25:  # probably wearing hat
            has_hair *= (
                (0.5 - hat) / 0.5 * has_hair
            )  # give a penalty to the hair confidence (hope this helps!)

        result = {
            "has_hair": has_hair,
            "hair_colour": hair_colour,
            "hair_shape": hair_shape,
            "facial_hair": facial_hair,
            "glasses": glasses,
            "hat": hat,
        }

        return result

reference_colors = {
    "red": [255, 0, 0],
    "green": [0, 255, 0],
    "blue": [0, 0, 255],
    "white": [255, 255, 255],
    "black": [0, 0, 0],
    "yellow": [255, 255, 0],
    "cyan": [0, 255, 255],
    "magenta": [255, 0, 255],
    "gray": [128, 128, 128],
    "orange": [255, 165, 0],
    "purple": [128, 0, 128],
    "brown": [139, 69, 19],
    "pink": [255, 182, 193],
    "beige": [245, 245, 220],
    "maroon": [128, 0, 0],
    "olive": [128, 128, 0],
    "navy": [0, 0, 128],
    "lime": [50, 205, 50],
    "golden": [255, 223, 0],
    "teal": [0, 128, 128],
    "coral": [255, 127, 80],
    "salmon": [250, 128, 114],
    "turquoise": [64, 224, 208],
    "violet": [238, 130, 238],
    "platinum": [229, 228, 226],
    "ochre": [204, 119, 34],
    "burntsienna": [233, 116, 81],
    "chocolate": [210, 105, 30],
    "tan": [210, 180, 140],
    "ivory": [255, 255, 240],
    "goldenrod": [218, 165, 32],
    "orchid": [218, 112, 214],
    "honey": [238, 220, 130],
    "lavender": [230, 230, 250],
    "mint": [189, 252, 201],
    "peach": [255, 229, 180],
    "ruby": [224, 17, 95],
    "indigo": [75, 0, 130],
    "amber": [255, 191, 0],
    "emerald": [80, 200, 120],
    "sapphire": [15, 82, 186],
    "aquamarine": [127, 255, 212],
    "periwinkle": [204, 204, 255],
    "fuchsia": [255, 0, 255],
    "raspberry": [227, 11, 92],
    "slate": [112, 128, 144],
    "charcoal": [54, 69, 79]
}

# reference_colors = {colour: arr.T for colour, arr in reference_colors.items()}

def rgb_to_lab(color):
    rgb_color = np.uint8([[color]])
    lab_color = cv2.cvtColor(rgb_color, cv2.COLOR_RGB2Lab)
    return lab_color[0][0]

def cie76_distance(color1, color2):
    lab1 = rgb_to_lab(color1)
    lab2 = rgb_to_lab(color2)
    return np.linalg.norm(lab1 - lab2)

def estimate_color(rgb_array):    
    # Calculate distances to each reference color
    # distances = {color: cie76_distance(rgb_array, ref_rgb) for color, ref_rgb in reference_colors.items()}
    distances = {color: np.linalg.norm(rgb_array - ref_rgb) for color, ref_rgb in reference_colors.items()}
    
    # Find the color with the smallest distance
    estimated_color = min(distances, key=distances.get)
    
    return estimated_color


def split_and_sample_colors(image, mask, square_size):
    height, width, _ = image.shape
    squares_colors = {}
    valid_squares = set()
    
    square_index = 0
    
    for y in range(0, height, square_size):
        for x in range(0, width, square_size):
            square = image[y:y + square_size, x:x + square_size]
            mask_square = mask[y:y + square_size, x:x + square_size]
            
            # Calculate the average color
            average_color = square.mean(axis=(0, 1))
            
            # Save the average color in the dictionary
            squares_colors[square_index] = average_color  # estimate_color(average_color)
            
            # Check the mask condition
            if np.sum(mask_square) > 0.5 * square_size * square_size:
                valid_squares.add(square_index)
            
            square_index += 1
    
    return squares_colors, valid_squares


def gaussian_blur(image, kernel_size, rep=3):
    """
    Apply Gaussian blur to an RGB image.

    Parameters:
    image (numpy.ndarray): The input RGB image.
    kernel_size (int): The size of the Gaussian kernel. If an even number is provided, it will be incremented to the next odd number.

    Returns:
    numpy.ndarray: The blurred RGB image.
    """
    if kernel_size % 2 == 0:
        kernel_size += 1  # Increment kernel size to make it odd if it's even
    
    for _ in range(rep):
        image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    return image


def visualize_grids(image, squares_colors, square_size):
    """
    Display the image with colored grids based on average colors.

    Parameters:
    image (numpy.ndarray): The original RGB image.
    squares_colors (dict): Dictionary with the average colors of each square.
    square_size (int): The size of each square.
    """
    height, width, _ = image.shape
    grid_image = np.zeros((height, width, 3), dtype=np.uint8)

    square_index = 0
    for y in range(0, height, square_size):
        for x in range(0, width, square_size):
            # color = np.array(reference_colors[squares_colors[square_index]], dtype=np.uint8)
            color = np.array(squares_colors[square_index])
            cv2.rectangle(grid_image, (x, y), (x + square_size, y + square_size), color.tolist(), -1)
            cv2.rectangle(grid_image, (x, y), (x + square_size, y + square_size), (255, 255, 255), 1)
            square_index += 1

    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(grid_image, cv2.COLOR_BGR2RGB))
    plt.title('Image with Average Color Grids')
    plt.axis('off')
    plt.show()


class ImageOfCloth(ImageWithMasksAndAttributes):
    def __init__(
        self,
        image: np.ndarray,
        masks: dict[str, np.ndarray],
        attributes: dict[str, float],
        categories_and_attributes: CategoriesAndAttributes,
    ):
        super().__init__(image, masks, attributes, categories_and_attributes)

    @classmethod
    def from_parent_instance(
        cls, parent_instance: ImageWithMasksAndAttributes
    ) -> "ImageOfCloth":
        """
        Creates an instance of ImageOfCloth using the properties of an
        instance of ImageWithMasksAndAttributes.
        """
        return cls(
            image=parent_instance.image,
            masks=parent_instance.masks,
            attributes=parent_instance.attributes,
            categories_and_attributes=parent_instance.categories_and_attributes,
        )

    def describe(self) -> dict:
        # Maybe not keep ‘sleeveless’ anymore but might do this in the actual environment.

        result = {
            "top": self.attributes["top"] / 2,
            "outwear": self.attributes["outwear"] / 2,
            "dress": self.attributes["dress"] / 2,
        }

        max_prob = 0.0
        max_attribute = "short sleeve top"
        for attribute in ["short sleeve top", "long sleeve top", "vest", "sling"]:
            if self.attributes[attribute] > max_prob:
                max_prob = self.attributes[attribute]
                max_attribute = attribute
        if max_attribute in ["vest", "sling"]:
            max_attribute = "sleeveless top"
        result["max_top"] = max_attribute

        max_prob = 0.0
        max_attribute = "short sleeve outwear"
        for attribute in [
            "short sleeve outwear",
            "long sleeve outwear",
        ]:
            if self.attributes[attribute] > max_prob:
                max_prob = self.attributes[attribute]
                max_attribute = attribute
        result["max_outwear"] = max_attribute

        max_prob = 0.0
        max_attribute = "short sleeve dress"
        for attribute in [
            "short sleeve dress",
            "long sleeve dress",
            "vest dress",
            "sling dress",
        ]:
            if self.attributes[attribute] > max_prob:
                max_prob = self.attributes[attribute]
                max_attribute = attribute
        if max_attribute in ["vest dress", "sling dress"]:
            max_attribute = "sleeveless dress"
        result["max_dress"] = max_attribute

        # QUICK FIX that won't break the code
        result["dress"] = 0.0

        # ========= colour estimation below: =========
        blurred_image = gaussian_blur(self.image, kernel_size=3, rep=3)
        for cloth in ["top", "down", "outwear", "dress"]:
            mask = self.masks[cloth]
            squares_colors, valid_squares = split_and_sample_colors(blurred_image, mask, 20)
            visualize_grids(blurred_image, squares_colors, square_size=20)
            pass

        return result
