import math
from function import FunctionManager
from regression import minimise_loss, find_classification
from lossfunction import squared_error
from plotting import plot_ideal_functions, plot_points_with_their_ideal_function
from utils import write_deviation_results_to_sqlite

# This constant is the factor for the criterion. It is specific to the assignment
ACCEPTED_FACTOR = math.sqrt(2)

if __name__ == '__main__':
    # Provide paths for csv files
    ideal_path = "data/ideal.csv"
    train_path = "data/train.csv"

    # The FunctionManager accepts a path to a csv and parses Function objects from the data.
    # A Function  stores X and Y points of a function. It uses Pandas to do this efficiently.
    candidate_ideal_function_manager = FunctionManager(path_of_csv=ideal_path)
    train_function_manager = FunctionManager(path_of_csv=train_path)

    # A FunctionManager uses the .to_sql function from Pandas
    # The suffix is added to comply to the requirement of the structure of the table
    train_function_manager.to_sql(file_name="training", suffix=" (training func)")
    candidate_ideal_function_manager.to_sql(file_name="ideal", suffix=" (ideal func)")

    # As Recap:
    # Within train_function_manager 4 functions are stored.
    # Withing ideal_function_manager 50 functions are stored.
    # In the next step we can use this data to compute an IdealFunction.
    # An IdealFunction amongst others stores best fitting function, the train data and is able to compute the tolerance.
    # All we now need to do is iterate over all train_functions
    # Matching ideal functions are stored in a list.
    ideal_functions = []
    for train_function in train_function_manager:
        # minimise_loss is able to compute the best fitting function given the train function
        ideal_function = minimise_loss(training_function=train_function,
                                       list_of_candidate_functions=candidate_ideal_function_manager.functions,
                                       loss_function=squared_error)
        ideal_function.tolerance_factor = ACCEPTED_FACTOR
        ideal_functions.append(ideal_function)

    # We can use the classification to do some plotting
    plot_ideal_functions(ideal_functions, "train_and_ideal")

    # Now it is time to look at all points within the test data
    # The FunctionManager provides all the necessary to load a CSV, so it will be reused.
    # Instead of multiple Functions like before, it will now contain a single "Function" at location [0]
    # The benefit is that we can iterate over each point with the Function object
    test_path = "data/test.csv"
    test_function_manager = FunctionManager(path_of_csv=test_path)
    test_function = test_function_manager.functions[0]

    points_with_ideal_function = []
    for point in test_function:
        ideal_function, delta_y = find_classification(point=point, ideal_functions=ideal_functions)
        result = {"point": point, "classification": ideal_function, "delta_y": delta_y}
        points_with_ideal_function.append(result)

    # Recap: within points_with_ideal_functions a list of dictionaries is stored.
    # These dictionaries represent the classification result of each point.

    # We can plot all the points with the corresponding classification function
    plot_points_with_their_ideal_function(points_with_ideal_function, "point_and_ideal")

    # Finally the dict object is used to write it to a sqlite
    # In this method a pure SQLAlchamy approach has been choosen with a MetaData object to save myself from SQL-Language
    write_deviation_results_to_sqlite(points_with_ideal_function)
    print("following files created:")
    print("training.db: All training functions as sqlite database")
    print("ideal.db: All ideal functions as sqlite database")
    print("mapping.db: Result of point test in which the ideal function and its delta is computed")
    print("train_and_ideal.html: View the train data as scatter and the best fitting ideal function as curve")
    print("points_and_ideal.html: View for those point with a matching ideal function the distance between them in a figure")

    print("Author: Maurice ten Koppel")
    print("Date: 01. September 2020")
    print("Script completed successfully")



