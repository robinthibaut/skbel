# Copyright (c) 2022. Maximilian Ramgraber, Massachusetts Institute of Technology, USA; Robin Thibaut,
# Ghent University, Belgium

import copy
import itertools

import numpy as np
from scipy.optimize import minimize, root
from sklearn.preprocessing import StandardScaler
from loguru import logger

__all__ = ["TransportMap"]


class TransportMap:
    def __init__(
        self,
        monotone: list,
        nonmonotone: list,
        X: np.array,
        polynomial_type: str = "hermite function",
        monotonicity: str = "integrated rectifier",
        standardize_samples: bool = True,
        workers: int = 1,
        ST_scale_factor: float = 1.0,
        ST_scale_mode: str = "dynamic",
        coeffs_init: float = 0.1,
        linearization: float = None,
        linearization_specified_as_quantiles: bool = True,
        linearization_increment: float = 1e-6,
        regularization: str = None,
        regularization_lambda: float = 0.1,
        quadrature_input: dict = None,
        rectifier_type: str = "exponential",
        delta: float = 0.0,
    ):
        """This toolbox contains functions required to construct, optimize, and
        evaluate transport methods.

         Maximilian Ramgraber, January 2022

        :param monotone: list specifying the structure of the monotone part of the transport map component functions.
        :param nonmonotone: list specifying the structure of the non-monotone part of the transport map component
            functions.
        :param X: N-by-D array of the training samples used to optimize the transport map, where N is the number of
            samples and D is the number of dimensions
        :param polynomial_type: keyword which specifies what kinds of polynomials are used for the transport map
            component functions.
        :param monotonicity: keyword which specifies through what method the transport map ensures monotonicity in the
            last dimensions. Must be 'integrated rectifier' or 'separable monotonicity'.
        :param standardize_samples: a True/False flag determining whether the transport map should standardize the
            training samples before optimization
        :param workers: number of workers for parallel optimization. If set to 1, parallelized optimization is inactive.
        :param ST_scale_factor: a float which scales the width of special terms used in the map components, such as
            'RBF 0', 'iRBF 0', 'LET 0', or 'RET 0'.
        :param ST_scale_mode: keyword which defines whether the width of special term scale factors is determined based
            on neighbouring special terms ('dynamic') or fixed as ST_scale_factor ('static').
        :param coeffs_init: value used to initialize the coefficients at the start of the map optimization.
        :param linearization: float which specifies boundary values used to linearize the map components in the tails.
             Its role is specified by linearization_specified_as_quantiles.
        :param linearization_specified_as_quantiles: flag which specifies whether the linearization thresholds are
            specified as quantiles (True) or absolute values (False). If True, boundaries are placed at linearization and
            1-linearization, if False, at linearization and -linearization. Only used if linearization is not None.
        :param linearization_increment: increment used for the linearization of the map component functions.
            Only used if linearization is not None.
        :param regularization: keyword which specifies if regularization is used, and if so, what kind of regularization
            ('L1' or 'L2').
        :param regularization_lambda: float which specifies the weight for the map coefficient regularization.
            Only used if regularization is not None.
        :param quadrature_input: dictionary for optional keywords to overwrite the default variables in the function
            Gauss_quadrature. Only used if monotonicity = 'integrated rectifier'.
        :param rectifier_type: keyword which specifies which function is used to rectify the monotone map components.
            Only used if monotonicity = 'integrated rectifier'.
        :param delta: small increment added to the rectifier to prevent numerical underflow.
            Only used if monotonicity = 'integrated rectifier'.
        """

        if quadrature_input is None:
            quadrature_input = {}

            # Load in pre-defined variables

        self.D = len(monotone)  # number of dimensions

        self.skip_dimensions = X.ndim - self.D  # number of dimensions to skip

        self.monotone = monotone  # monotonicity constraints
        self.nonmonotone = nonmonotone  # non-monotonicity constraints

        self.workers = workers  # number of workers for parallelization

        self.rectifier_type = rectifier_type  # type of rectifier used
        self.delta = delta  # small increment added to rectifier
        self.rect = self.Rectifier(
            mode=self.rectifier_type, delta=self.delta
        )  # rectifier function

        self.quadrature_input = quadrature_input

        self.ST_scale_factor = ST_scale_factor
        self.ST_scale_mode = ST_scale_mode

        self.coeffs_init = coeffs_init

        self.verbose = False

        self.regularization = regularization
        self.regularization_lambda = regularization_lambda

        self.linearization = linearization
        self.linearization_specified_as_quantiles = linearization_specified_as_quantiles
        self.linearization_increment = linearization_increment

        self.monotonicity = monotonicity
        if self.monotonicity.lower() not in [
            "integrated rectifier",
            "separable monotonicity",
        ]:
            raise ValueError(
                "'monotonicity' type "
                + str(self.monotonicity)
                + " not understood. "
                + "Must be either 'integrated rectifier' or 'separable monotonicity'."
            )

        if self.ST_scale_mode not in ["dynamic", "static"]:
            raise ValueError("'ST_scale_mode' must be either 'dynamic' or 'static'.")

        # Read and assign the polynomial type
        self.polynomial_type = polynomial_type

        # Determine the derivative and polynomial terms depending on the chosen type
        if (
            polynomial_type.lower() == "standard"
            or polynomial_type.lower() == "polynomial"
            or polynomial_type.lower() == "power series"
        ):
            self.polyfunc = np.polynomial.polynomial.Polynomial
            self.polyfunc_der = np.polynomial.polynomial.polyder
            self.polyfunc_str = "np.polynomial.Polynomial"
        elif (
            polynomial_type.lower() == "hermite"
            or polynomial_type.lower() == "phycisist's hermite"
            or polynomial_type.lower() == "phycisists hermite"
        ):
            self.polyfunc = np.polynomial.hermite.Hermite
            self.polyfunc_der = np.polynomial.hermite.hermder
            self.polyfunc_str = "np.polynomial.Hermite"
        elif (
            polynomial_type.lower() == "hermite_e"
            or polynomial_type.lower() == "probabilist's hermite"
            or polynomial_type.lower() == "probabilists hermite"
        ):
            self.polyfunc = np.polynomial.hermite_e.HermiteE
            self.polyfunc_der = np.polynomial.hermite_e.hermeder
            self.polyfunc_str = "np.polynomial.HermiteE"
        elif polynomial_type.lower() == "chebyshev":
            self.polyfunc = np.polynomial.chebyshev.Chebyshev
            self.polyfunc_der = np.polynomial.chebyshev.chebder
            self.polyfunc_str = "np.polynomial.Chebyshev"
        elif polynomial_type.lower() == "laguerre":
            self.polyfunc = np.polynomial.laguerre.Laguerre
            self.polyfunc_der = np.polynomial.laguerre.lagder
            self.polyfunc_str = "np.polynomial.Laguerre"
        elif polynomial_type.lower() == "legendre":
            self.polyfunc = np.polynomial.legendre.Legendre
            self.polyfunc_der = np.polynomial.legendre.legder
            self.polyfunc_str = "np.polynomial.Legendre"
        elif (
            polynomial_type.lower() == "hermite function"
            or polynomial_type.lower() == "hermite_function"
            or polynomial_type.lower() == "hermite functions"
        ):
            self.polynomial_type = "hermite function"  # Unify this polynomial string, so we can use it as a flag
            self.polyfunc = np.polynomial.hermite_e.HermiteE
            self.polyfunc_der = np.polynomial.hermite_e.hermeder
            self.polyfunc_str = "np.polynomial.HermiteE"
        else:
            raise Exception(
                "Polynomial type not understood. The variable polynomial_type should be either 'power series', "
                "'hermite', 'hermite_e', 'chebyshev', 'laguerre', or 'legendre'. "
            )

        # Load and prepare the variables

        # Load and standardize the samples
        self.X = X
        self.standardize_samples = standardize_samples
        if self.standardize_samples:
            self.scaler = StandardScaler()
            self.X = self.scaler.fit_transform(X)

            # Construct the monotone and non-monotone functions

        # The function_constructor yields six variables:
        #   - fun_mon               : list of monotone functions
        #   - fun_mon_strings       : list of monotone function strings
        #   - coeffs_mon            : list of coefficients for monotone function
        #   - fun_nonmon            : list of non-monotone functions
        #   - fun_nonmon_strings    : list of non-monotone function strings
        #   - coeffs_nonmon         : list of coefficients for non-monotone function

        self.function_constructor_alternative()

        # Precalculate the Psi matrices

        # The function_constructor yields two variables:
        #   - Psi_mon               : list of monotone basis evaluations
        #   - Psi_nonmon            : list of non-monotone basis evaluations

        self.precalculate()

    def check_inputs(self):

        """This function runs some preliminary checks on the input provided,
        alerting the user to any possible input errors."""

        if self.monotonicity.lower() not in [
            "integrated rectifier",
            "separable monotonicity",
        ]:
            raise ValueError(
                "'monotonicity' type "
                + str(self.monotonicity)
                + " not understood. "
                + "Must be either 'integrated rectifier' or 'separable monotonicity'."
            )

        if self.ST_scale_mode not in ["dynamic", "static"]:
            raise ValueError("'ST_scale_mode' must be either 'dynamic' or 'static'.")

        if (
            self.hermite_function_threshold_mode != "composite"
            and self.hermite_function_threshold_mode != "individual"
        ):
            raise ValueError(
                "The flag hermite_function_threshold_mode must be "
                + "'composite' or 'individual'. Currently, it is defined as "
                + str(self.hermite_function_threshold_mode)
            )

        if self.regularization is not None:

            if self.monotonicity.lower() != "separable monotonicity":

                if self.regularization.lower() not in ["l2"]:
                    raise ValueError(
                        "When using 'separable monotonicity',"
                        + "'regularization' must either be None "
                        + "(deactivated) or 'L2' (L2 regularization). Currently, it "
                        + "is defined as "
                        + str(self.regularization)
                    )
            else:

                if self.regularization.lower() not in ["l2"]:
                    raise ValueError(
                        "When using 'integrated rectifier',"
                        + "'regularization' must either be None "
                        + "(deactivated), 'L1' (L1 regularization) or "
                        + "'L2' (L2 regularization). Currently, it "
                        + "is defined as "
                        + str(self.regularization)
                    )

    def reset(self, X: np.array):

        """This function is used if the transport map has been initiated with a
        different set of samples. It resets the standardization variables and
        the map's coefficients, requiring new optimization.

        :param X: N-by-D array of the training samples used to optimize the transport map, where N is the number of
            samples and D is the number of dimensions
        """

        if len(X.shape) != 2:
            raise Exception(
                "X should be a two-dimensional array of shape (N,D), N = number of samples, D = number of dimensions. "
                "Current shape of X is " + str(X.shape)
            )

        self.X = X

        # Standardize the samples, if desired
        if self.standardize_samples:
            self.scaler = StandardScaler()
            self.X = self.scaler.fit_transform(X)

        # Set all parameters to zero
        for k in range(self.D):
            self.coeffs_mon[k] *= 0
            self.coeffs_nonmon[k] *= 0

            self.coeffs_mon[k] += self.coeffs_init  # 1E-3
            self.coeffs_nonmon[k] += self.coeffs_init  # 1E-3

        # Precalculate the Psi matrices
        self.precalculate()

    def precalculate(self):

        """This function pre-calculates matrices of basis function evaluations
        for the samples provided.

        These matrices can be used to optimize the maps more quickly.
        """

        # Prepare precalculation matrices
        self.Psi_mon = []
        self.Psi_nonmon = []

        # Precalculate matrices
        for k in range(self.D):
            # Pre-allocate empty matrices
            self.Psi_mon.append(self.fun_mon[k](self.X, self))
            self.Psi_nonmon.append(self.fun_nonmon[k](self.X, self))

        # Precalculate locations of any special terms
        self.calculate_special_term_locations()

    def write_basis_function(
        self, term: list or str, mode: str = "standard", k: int = None
    ):

        """This function assembles a string for a specific term of the map
        component functions. This can be a polynomial, a Hermite function, a
        radial basis function, or similar.

        :param term: either an empty list, a list, or a string, which specifies what function type this term is supposed
            to be. Empty lists correspond to a constant, lists specify polynomials or Hermite functions, and strings
            denote special terms such as radial basis functions.
        :param mode: a keyword which defines whether the term's string returned should be conventional ('standard')
            or its derivative ('derivative').
        :param k: an integer specifying what dimension of the samples the 'term' corresponds to. Used to clarify with
            respect to what variable we take the derivative.
        """

        # First, check for input errors

        # Check if mode is valid
        if mode not in ["standard", "derivative"]:
            raise ValueError(
                "Mode must be either 'standard' or 'derivative'. Unrecognized mode: "
                + str(mode)
            )

        # If derivative, check if k is specified
        if mode == "derivative" and k is None:
            raise ValueError(
                "If mode == 'derivative', specify an integer for k to inform with regards to which variable we take "
                "the derivative. Currently specified: k = " + str(k)
            )

        # If derivative, check if k is an integer
        if mode == "derivative" and type(k) is not int:
            raise ValueError(
                "If mode == 'derivative', specify an integer for k to inform with regards to which variable we take "
                "the derivative. Currently specified: k = " + str(k)
            )

        # Initiate the modifier log

        # This variable returns information about whether there is anything
        # special about this term. If this is not None, it is a dictionary with
        # the following possible keys:
        #   "constant"  this is a constant term
        #   "ST"        this is a RBF-based special term
        #   "HF"        this is a polynomial with a Hermite function modifier
        #   "LIN"       this is a polynomial with a linearization modifier
        #   "HFLIN"     this is a polynomial with both modifiers

        modifier_log = {}

        # Constant
        # If the entry is an empty list, add a constant
        if not term:

            if mode == "standard":

                # Construct the string
                string = "np.ones(__x__.shape[:-1])"

            elif mode == "derivative":

                # Construct the string
                string = "np.zeros(__x__.shape[:-1])"

            # Log this term
            modifier_log = {"constant": None}

        # Special term

        # If the entry is a string, it denotes a special term
        elif type(term) == str:

            # Split the string
            STtype, i = term.split(" ")

            # Log this term
            modifier_log = {"ST": int(i)}

            # Left edge term

            if STtype.lower() == "let":  # The special term is a left edge term

                if mode == "standard":

                    # Construct the string
                    string = (
                        "((__x__[...,"
                        + i
                        + "] - __mu__)*(1-scipy.special.erf((__x__[...,"
                        + i
                        + "] - __mu__)/(np.sqrt(2)*__scale__))) - __scale__*np.sqrt(2/np.pi)*np.exp(-((__x__[...,"
                        + i
                        + "] - __mu__)/(np.sqrt(2)*__scale__))**2))/2"
                    )

                elif mode == "derivative":

                    # Construct the string
                    if int(i) == k:

                        string = (
                            "(1 - scipy.special.erf((__x__[...,"
                            + i
                            + "] - __mu__)/(np.sqrt(2)*__scale__)))/2"
                        )

                    else:

                        string = "np.zeros(__x__.shape[:-1])"

            # Right edge term

            elif STtype.lower() == "ret":  # The special term is a right edge term

                if mode == "standard":

                    # Construct the string
                    string = (
                        "((__x__[...,"
                        + i
                        + "] - __mu__)*(1+scipy.special.erf((__x__[...,"
                        + i
                        + "] - __mu__)/(np.sqrt(2)*__scale__))) + __scale__*np.sqrt(2/np.pi)*np.exp(-((__x__[...,"
                        + i
                        + "] - __mu__)/(np.sqrt(2)*__scale__))**2))/2"
                    )

                elif mode == "derivative":

                    # Construct the string
                    if int(i) == k:

                        string = (
                            "(1 + scipy.special.erf((__x__[...,"
                            + i
                            + "] - __mu__)/(np.sqrt(2)*__scale__)))/2"
                        )

                    else:

                        string = "np.zeros(__x__.shape[:-1])"

            # Radial basis function

            elif STtype.lower() == "rbf":  # The special term is a right edge term

                if mode == "standard":

                    # Construct the string
                    string = (
                        "1/(np.sqrt(2*np.pi)*__scale__)*np.exp(-(__x__[...,"
                        + i
                        + "] - __mu__)**2/(2*__scale__**2))"
                    )

                elif mode == "derivative":

                    # Construct the string
                    if int(i) == k:

                        string = (
                            "(__x__[...,"
                            + i
                            + "] - __mu__)/(np.sqrt(2*np.pi)*__scale__**3)*np.exp(-(__x__[...,"
                            + i
                            + "] - __mu__)**2/(2*__scale__**2))"
                        )

                    else:

                        string = "np.zeros(__x__.shape[:-1])"

            # Integrated radial basis function

            elif STtype.lower() == "irbf":  # The special term is a right edge term

                if mode == "standard":

                    # Construct the string
                    string = (
                        "(1 + scipy.special.erf((__x__[...,"
                        + i
                        + "] - __mu__)/(np.sqrt(2)*__scale__)))/2"
                    )

                elif mode == "derivative":

                    # Construct the string
                    if int(i) == k:

                        string = (
                            "1/(np.sqrt(2*np.pi)*__scale__)*np.exp(-(__x__[...,"
                            + i
                            + "] - __mu__)**2/(2*__scale__**2))"
                        )

                    else:

                        string = "np.zeros(__x__.shape[:-1])"

            # Unrecognized special term

            else:

                raise ValueError(
                    "Special term '"
                    + str(STtype)
                    + "' not "
                    + "understood. Currently, only LET, RET, iRBF, and RBF "
                    + "are implemented."
                )

        # Polynomial term

        # Otherwise, it is a standard polynomial term
        else:

            # Check for modifiers

            # Check for Hermite function modifier
            # TRUE if modifier is active, else FALSE
            hermite_function_modifier = np.asarray(
                [True if i == "HF" else False for i in term], dtype=bool
            ).any()

            # Check for linearization modifier
            # TRUE if modifier is active, else FALSE
            linearize = np.asarray(
                [True if i == "LIN" else False for i in term], dtype=bool
            ).any()

            # Check if linearization is also activated
            if linearize and self.linearization is None:
                raise Exception(
                    "'LIN' modifier specified in variable monotone, but the variable linearization is defined as "
                    "None. Please specify a scalar linearization or remove the 'LIN' modifier. "
                )

            # Remove all string-based modifiers
            term = [i for i in term if type(i) != str]

            # Construct the polynomial term

            # Extract the unique entries and their counts
            ui, ct = np.unique(term, return_counts=True)

            # Both Hermite function and linearization modifiers are active
            if hermite_function_modifier and linearize:

                # Log this term
                modifier_log = {"HFLIN": None}

            # Hermite function modifiers is active
            elif hermite_function_modifier:

                # Log this term
                modifier_log = {"HF": None}

            # Linearization modifiers is active
            elif linearize:

                # Log this term
                modifier_log = {"LIN": None}

            # Add a "variables" key to the modifier_log, if it does not already exist
            if "variables" not in list(modifier_log.keys()):
                modifier_log["variables"] = {}

            # Create an empty string
            string = ""

            # Go through all unique entries
            for i in range(len(ui)):

                # Create an array of polynomial coefficients
                dummy_coefficients = [0.0] * ct[i] + [1.0]

                # Normalize the influence of Hermite functions
                if hermite_function_modifier:
                    # Evaluate a naive Hermite function
                    hf_x = np.linspace(-100, 100, 100001)
                    hfeval = self.polyfunc(dummy_coefficients)(hf_x) * np.exp(
                        -(hf_x ** 2) / 4
                    )

                    # Scale the polynomial coefficient to normalize its maximum value
                    dummy_coefficients[-1] = 1 / np.max(np.abs(hfeval))

                # Standard polynomial

                if mode == "standard" or (mode == "derivative" and ui[i] != k):

                    # Create a variable key
                    key = "P_" + str(ui[i]) + "_O_" + str(ct[i])
                    if hermite_function_modifier:
                        key += "_HF"
                    if linearize:
                        key += "_LIN"

                    # Set up function
                    # Extract the polynomial
                    var = self.polyfunc_str

                    # Open outer parenthesis
                    var += "(["

                    # Add polynomial coefficients
                    for dc in dummy_coefficients:
                        var += str(dc) + ","

                    # Remove the last ","
                    var = var[:-1]

                    # Close outer parenthesis
                    var += "])"

                    # Add variable

                    var += "(__x__[...," + str(ui[i]) + "])"

                    # Add Hermite function

                    if hermite_function_modifier:
                        var += "*np.exp(-__x__[...," + str(ui[i]) + "]**2/4)"

                    # Save the variable
                    if key not in list(modifier_log["variables"].keys()):
                        modifier_log["variables"][key] = var

                    # Add the variable to the string
                    string += key

                    # Add a multiplier, in case there are more terms
                    string += " * "

                # Derivative of polynomial

                elif mode == "derivative":

                    # Create a variable key for the standard polynomial
                    key = "P_" + str(ui[i]) + "_O_" + str(ct[i])

                    # Create a variable key for its derivative
                    keyder = "P_" + str(ui[i]) + "_O_" + str(ct[i]) + "_DER"

                    # Set up function

                    # Find the derivative coefficients
                    dummy_coefficients_der = self.polyfunc_der(dummy_coefficients)

                    # Extract the polynomial
                    varder = self.polyfunc_str

                    # Open outer parenthesis
                    varder += "(["

                    # Add polynomial coefficients
                    for dc in dummy_coefficients_der:
                        varder += str(dc) + ","

                    # Remove the last ","
                    varder = varder[:-1]

                    # Close outer parenthesis
                    varder += "])"

                    # Add variable

                    varder += "(__x__[...," + str(ui[i]) + "])"

                    # Save the variable
                    if keyder not in list(modifier_log["variables"].keys()):
                        modifier_log["variables"][keyder] = varder

                    # Add the variable to the string
                    if not hermite_function_modifier:
                        string += varder

                    # Add Hermite function

                    if hermite_function_modifier:

                        # If we have a hermite function modifier, we also need
                        # the original form of the polynomial

                        # Set up function
                        # Extract the polynomial
                        varbase = self.polyfunc_str

                        # Open outer parenthesis
                        varbase += "(["

                        # Add polynomial coefficients
                        for dc in dummy_coefficients:
                            varbase += str(dc) + ","

                        # Remove the last ","
                        varbase = varbase[:-1]

                        # Close outer parenthesis
                        varbase += "])"

                        # Add variable

                        varbase += "(__x__[...," + str(ui[i]) + "])"

                        # Save the variable -
                        if key not in list(modifier_log["variables"].keys()):
                            modifier_log["variables"][key] = varbase

                        # Now we can construct the actual derivative

                        string = (
                            "-1/2*np.exp(-__x__[...,"
                            + str(ui[i])
                            + "]**2/4)*(__x__[...,"
                            + str(ui[i])
                            + "]*"
                            + key
                            + " - 2*"
                            + keyder
                            + ")"
                        )

                    # Add a multiplier, in case there are more terms
                    string += " * "

            # Remove the last multiplier " * "
            string = string[:-3]

            # If the variable we take the derivative against is not in the term,
            # overwrite the string with zeros
            if mode == "derivative" and k not in ui:
                # Overwrite string with zeros
                string = "np.zeros(__x__.shape[:-1])"

        return string, modifier_log

    def function_constructor_alternative(self):

        """This function assembles the string for the monotone and non-monotone
        map components, then converts these strings into functions."""

        self.fun_mon = []  # Initialize the list of monotone functions
        self.fun_mon_strings = []  # Initialize the list of monotone function strings
        self.coeffs_mon = []  # Initialize the list of monotone coefficients

        self.fun_nonmon = []  # Initialize the list of non-monotone functions
        self.fun_nonmon_strings = (
            []
        )  # Initialize the list of non-monotone function strings
        self.coeffs_nonmon = []  # Initialize the list of non-monotone coefficients

        # Find out how many function terms we are building
        K = len(self.monotone)

        # Check for any special terms
        self.check_for_special_terms()
        self.calculate_special_term_locations()

        # Check if the monotone term is linear
        self.monotone_term_is_linear = []

        # Go through all terms
        for k in range(K):

            # Step 1: Build the monotone function

            # Define modules to load
            modules = [" ", ""]

            # Check if this term is linear
            if (
                len(self.monotone[k]) == 1
                and type(self.monotone[k][0]) != str
                and len(self.monotone[k][0]) != 0
                and self.monotonicity == "separable monotonicity"
            ):

                # Criteria:
                #   - Monotone term has at most one entry
                #   - this entry is not a string (special term)
                #   - this entry is not empty (i.e., not [] ) (a constant)
                #   - the monotonicity mode is 'separable monotonicity'
                # This is relevant because we use this information to use
                # an alternative, faster, analytical root finding if the
                # monotone term is linear in x and linear in the coefficients.
                self.monotone_term_is_linear.append(True)

            else:

                self.monotone_term_is_linear.append(False)

            # Extract the terms

            # Define the terms composing the transport map component
            terms = []

            # Prepare a counter for the special terms
            ST_counter = np.zeros(self.X.shape[-1], dtype=int)

            # Prepare a dictionary for precalculated variables
            dict_precalc = {}

            # Mark which of these are special terms, in case we want to create
            # permutations of multiple RBFS
            ST_indices = []

            # Go through all terms
            for i, entry in enumerate(self.monotone[k]):

                # Convert the map specification to a function

                # Find the term's function
                term, modifier_log = self.write_basis_function(
                    term=entry, mode="standard"
                )

                # Extract any precalculations, where applicable

                # If this term includes and precalculations, extract them
                if "variables" in list(modifier_log.keys()):

                    # There are precalculating variables. Go through each
                    for key in list(modifier_log["variables"].keys()):

                        # Have we logged this one already?
                        if key not in list(dict_precalc.keys()):

                            # No, we haven't. Add it.
                            dict_precalc[key] = modifier_log["variables"][key].replace(
                                "__x__", "x"
                            )

                            # Wait a moment! Are we linearizing this term?
                            if key.endswith("_LIN"):
                                # Yes, we are! What dimension is this?
                                d = int(key.split("_")[1])

                                # Edit the term
                                dict_precalc[key] = (
                                    dict_precalc[key].replace("__x__", "x_trc")
                                    + " * "
                                    + "(1 - vec[:,"
                                    + str(d)
                                    + "]/"
                                    + str(self.linearization_increment)
                                    + ") + "
                                    + dict_precalc[key].replace("__x__", "x_ext")
                                    + " * "
                                    + "vec[:,"
                                    + str(d)
                                    + "]/"
                                    + str(self.linearization_increment)
                                )

                                # Post-processing for special terms

                # Is this term a special term?
                if "ST" in list(modifier_log.keys()):

                    # Mark this term as a special one
                    ST_indices.append(i)

                    # Yes, it is. Add additional modules to load, if necessary
                    if "import scipy.special" not in modules:
                        modules.append("import scipy.special")

                    # Extract this special term's dimension
                    idx = modifier_log["ST"]

                    # Replace __mu__ with the correct ST location variable
                    term = term.replace(
                        "__mu__",
                        "self.ST_centers_m["
                        + str(k)
                        + "]["
                        + str(idx)
                        + "]["
                        + str(ST_counter[idx])
                        + "]",
                    )

                    # Replace __scale__ with the correct ST location variable
                    term = term.replace(
                        "__scale__",
                        "self.ST_scales_m["
                        + str(k)
                        + "]["
                        + str(idx)
                        + "]["
                        + str(ST_counter[idx])
                        + "]",
                    )

                    # Increment the special term counter
                    ST_counter[idx] += 1

                    # Add the term to the list

                # If any dummy __x__ remain, replace them
                term = term.replace("__x__", "x")

                # Store the term
                terms.append(term)

                # If there are special cross-terms, create them

            # Are there multiple special terms?
            if (
                np.sum([True if x != 0 else False for x in self.RBF_counter_m[k, :]])
                > 1
            ):

                # Yes, there are multiple special terms. Extract these terms.
                RBF_terms = [terms[i] for i in ST_indices]

                # Check what variables these terms are affiliated with
                RBF_terms_dim = -np.ones(len(RBF_terms), dtype=int)
                for ki in range(k + 1 + self.skip_dimensions):
                    for i, term in enumerate(RBF_terms):
                        if "x[...," + str(ki) + "]" in term:
                            RBF_terms_dim[i] = ki
                RBF_terms_dims = np.unique(np.asarray(RBF_terms_dim))

                # Create a dictionary with the different terms
                RBF_terms_dict = {}
                for i in RBF_terms_dims:
                    RBF_terms_dict[i] = [
                        RBF_terms[j]
                        for j in range(len(RBF_terms))
                        if RBF_terms_dim[j] == i
                    ]

                # Create all combinations of terms
                RBF_terms_grid = copy.deepcopy(RBF_terms_dict[RBF_terms_dims[0]])
                for i in RBF_terms_dims[1:]:
                    # Create a grid with the next dimension
                    RBF_terms_grid = list(
                        itertools.product(
                            RBF_terms_grid, copy.deepcopy(RBF_terms_dict[i])
                        )
                    )

                    # Convert this list of tuples into a new list of strings
                    RBF_terms_grid = [
                        entry[0] + "*" + entry[1] for entry in RBF_terms_grid
                    ]

                # Now remove all original RBF terms
                terms = [entry for i, entry in enumerate(terms) if i not in ST_indices]

                # Now add all the grid terms
                terms += RBF_terms_grid

                # Add monotone coefficients

            # Append the parameters
            self.coeffs_mon.append(np.ones(len(terms)) * self.coeffs_init)

            # Assemble the monotone function

            # Prepare the basis string
            string = "def fun(x,self):\n\t\n\t"

            # Load module requirements

            for entry in modules:
                string += entry + "\n\t"
            string += "\n\t"  # Another line break for legibility

            # Prepare linearization, if necessary

            # If linearization is active, truncate the input x
            if self.linearization is not None:
                # First, find our which parts are outside the linearization hypercube
                string += "vec_below = copy.copy(x) - self.linearization_threshold[:,0][np.newaxis,:];\n\t"
                string += (
                    "vec_below[vec_below >= 0] = 0;\n\t"  # Set all values above to zero
                )
                string += "vec_above = copy.copy(x) - self.linearization_threshold[:,1][np.newaxis,:];\n\t"
                string += (
                    "vec_above[vec_above <= 0] = 0;\n\t"  # Set all values below to zero
                )
                string += "vec = vec_above + vec_below;\n\t"

                # Then convert the two arrays to boolean markers
                string += "below = (vec_below < 0);\n\t"  # Find all particles BELOW the lower linearization band
                string += "above = (vec_above > 0);\n\t"  # Find all particles ABOVE the upper linearization band
                string += "shift = np.asarray(below,dtype=float) + np.asarray(above,dtype=float);\n\t"  # This is a
                # matrix where all entries outside the linearization bands are 1 and all entries inside are 0

                # Truncate all values outside the hypercube
                string += "x_trc = copy.copy(x);\n\t"
                string += "for d in range(x.shape[1]):\n\t\t"
                string += "x_trc[below[:,d],d] = self.linearization_threshold[d,0];\n\t\t"  # All values below the
                # linearization band of this dimension are snapped to its border
                string += "x_trc[above[:,d],d] = self.linearization_threshold[d,1];\n\t"  # All values above the
                # linearization band of this dimension are snapped to its border

                # Add a space to the next block
                string += "\n\t"

                # Also crate an extrapolated version of x_trc
                string += "x_ext = copy.copy(x_trc);\n\t"
                string += (
                    "x_ext += shift*" + str(self.linearization_increment) + ";\n\t"
                )  # Offset all values which have been snapped by a small increment

                # Add a space to the next block
                string += "\n\t"

                # Prepare precalculated variables

            # Add all precalculation terms
            for key in list(dict_precalc.keys()):
                string += key + " = " + copy.copy(dict_precalc[key]) + ";\n\t"

                # Assemble function output

            # Prepare the result string
            if len(terms) == 1:  # Only a single term, no need for stacking

                string += "result = " + copy.copy(terms[0]) + "[:,np.newaxis];\n\t\n\t"

            else:  # If we have more than one term, start stacking the result

                # Prepare the stack
                string += "result = np.stack((\n\t\t"

                # Go through each entry in terms, add them one by one
                for entry in terms:
                    string += copy.copy(entry) + ",\n\t\t"

                # Remove the last ",\n\t\t" and close the stack
                string = string[:-4]
                string += "),axis=-1)\n\t\n\t"

            # Return the result
            string += "return result"

            # Finish function construction

            # Append the function string
            self.fun_mon_strings.append(string)

            # Create an actual function
            funstring = "fun_mon_" + str(k)
            exec(string.replace("fun", funstring), globals())
            exec("self.fun_mon.append(copy.deepcopy(" + funstring + "))")

            # Step 2: Build the non-monotone function

            # Append the parameters
            self.coeffs_nonmon.append(
                np.ones(len(self.nonmonotone[k])) * self.coeffs_init
            )

            # Define modules to load
            modules = [" ", ""]

            # Extract the terms

            # Define the terms composing the transport map component
            terms = []

            # Prepare a counter for the special terms
            ST_counter = np.zeros(self.X.shape[-1], dtype=int)

            # Prepare a dictionary for precalculated variables
            dict_precalc = {}

            # Go through all terms
            for entry in self.nonmonotone[k]:

                # Convert the map specification to a function

                # Find the term's function
                term, modifier_log = self.write_basis_function(
                    term=entry, mode="standard"
                )

                # Extract any precalculations, where applicable

                # If this term includes and precalculations, extract them
                if "variables" in list(modifier_log.keys()):

                    # There are precalculating variables. Go through each
                    for key in list(modifier_log["variables"].keys()):

                        # Have we logged this one already?
                        if key not in list(dict_precalc.keys()):

                            # No, we haven't. Add it.
                            dict_precalc[key] = copy.copy(
                                modifier_log["variables"][key]
                            ).replace("__x__", "x")

                            # Wait a moment! Are we linearizing this term?
                            if key.endswith("_LIN"):
                                # Yes, we are! What dimension is this?
                                d = int(copy.copy(key).split("_")[1])

                                # Edit the term
                                dict_precalc[key] = (
                                    copy.copy(dict_precalc[key]).replace(
                                        "__x__", "x_trc"
                                    )
                                    + " * "
                                    + "(1 - vec[:,"
                                    + str(d)
                                    + "]/"
                                    + str(self.linearization_increment)
                                    + ") + "
                                    + copy.copy(dict_precalc[key]).replace(
                                        "__x__", "x_ext"
                                    )
                                    + " * "
                                    + "vec[:,"
                                    + str(d)
                                    + "]/"
                                    + str(self.linearization_increment)
                                )

                                # Post-processing for special terms

                # Is this term a special term?
                if "ST" in list(modifier_log.keys()):

                    # Yes, it is. Add additional modules to load, if necessary
                    if "import scipy.special" not in modules:
                        modules.append("import scipy.special")

                    # Extract this special term's dimension
                    idx = modifier_log["ST"]

                    # Replace __mu__ with the correct ST location variable
                    term = term.replace(
                        "__mu__",
                        "self.ST_centers_nm["
                        + str(k)
                        + "]["
                        + str(idx)
                        + "]["
                        + str(ST_counter[idx])
                        + "]",
                    )

                    # Replace __scale__ with the correct ST location variable
                    term = term.replace(
                        "__scale__",
                        "self.ST_scales_nm["
                        + str(k)
                        + "]["
                        + str(idx)
                        + "]["
                        + str(ST_counter[idx])
                        + "]",
                    )

                    # Increment the special term counter
                    ST_counter[idx] += 1

                    # Add the term to the list

                # If any dummy __x__ remain, replace them
                term = term.replace("__x__", "x")

                # Store the term
                terms.append(copy.copy(term))

            # Assemble the monotone function

            # Only assemble the function if there actually is a non-monotone term
            if len(self.nonmonotone[k]) > 0:

                # Prepare the basis string
                string = "def fun(x,self):\n\t\n\t"

                # Load module requirements

                for entry in modules:
                    string += copy.copy(entry) + "\n\t"
                string += "\n\t"  # Another line break for legibility

                # Prepare linearization, if necessary

                # If linearization is active, truncate the input x
                if self.linearization is not None:
                    # First, find our which parts are outside the linearization hypercube
                    string += "vec_below = copy.copy(x) - self.linearization_threshold[:,0][np.newaxis,:];\n\t"
                    string += "vec_below[vec_below >= 0] = 0;\n\t"  # Set all values above to zero
                    string += "vec_above = copy.copy(x) - self.linearization_threshold[:,1][np.newaxis,:];\n\t"
                    string += "vec_above[vec_above <= 0] = 0;\n\t"  # Set all values below to zero
                    string += "vec = vec_above + vec_below;\n\t"

                    # Then convert the two arrays to boolean markers
                    string += "below = (vec_below < 0);\n\t"  # Find all particles BELOW the lower linearization band
                    string += "above = (vec_above > 0);\n\t"  # Find all particles ABOVE the upper linearization band
                    string += "shift = np.asarray(below,dtype=float) + np.asarray(above,dtype=float);\n\t"  # This is
                    # a matrix where all entries outside the linearization bands are 1 and all entries inside are 0

                    # Truncate all values outside the hypercube
                    string += "x_trc = copy.copy(x);\n\t"
                    string += "for d in range(x.shape[1]):\n\t\t"
                    string += "x_trc[below[:,d],d] = self.linearization_threshold[d,0];\n\t\t"  # All values below
                    # the linearization band of this dimension are snapped to its border
                    string += "x_trc[above[:,d],d] = self.linearization_threshold[d,1];\n\t"  # All values above the
                    # linearization band of this dimension are snapped to its border

                    # Add a space to the next block
                    string += "\n\t"

                    # Also crate an extrapolated version of x_trc
                    string += "x_ext = copy.copy(x_trc);\n\t"
                    string += (
                        "x_ext += shift*" + str(self.linearization_increment) + ";\n\t"
                    )  # Offset all values which have been snapped by a small increment

                    # Add a space to the next block
                    string += "\n\t"

                    # Prepare precalculated variables

                # Add all precalculation terms
                for key in list(dict_precalc.keys()):
                    string += key + " = " + copy.copy(dict_precalc[key]) + ";\n\t"

                    # Assemble function output

                # Prepare the result string
                if len(terms) == 1:  # Only a single term, no need for stacking

                    string += (
                        "result = " + copy.copy(terms[0]) + "[:,np.newaxis];\n\t\n\t"
                    )

                else:  # If we have more than one term, start stacking the result

                    # Prepare the stack
                    string += "result = np.stack((\n\t\t"

                    # Go through each entry in terms, add them one by one
                    for entry in terms:
                        string += copy.copy(entry) + ",\n\t\t"

                    # Remove the last ",\n\t\t" and close the stack
                    string = string[:-4]
                    string += "),axis=-1)\n\t\n\t"

                # Return the result
                string += "return result"

                # Finish function construction

                # Append the function string
                self.fun_nonmon_strings.append(string)

                # Create an actual function
                funstring = "fun_nonmon_" + str(k)
                exec(string.replace("fun", funstring), globals())
                exec("self.fun_nonmon.append(copy.deepcopy(" + funstring + "))")

            else:  # There are NO non-monotone terms

                # Create a function which just returns None
                string = "def fun(x,self):\n\t"
                string += "return None"

                # Append the function string
                self.fun_nonmon_strings.append(string)

                # Create an actual function
                funstring = "fun_nonmon_" + str(k)
                exec(string.replace("fun", funstring), globals())
                exec("self.fun_nonmon.append(copy.deepcopy(" + funstring + "))")

        # Step 3: Finalize

        # If monotonicity mode is 'separable monotonicity', we also require the
        # derivative of the monotone part of the map
        if self.monotonicity.lower() == "separable monotonicity":
            self.function_derivative_constructor_alternative()

        return

    def function_derivative_constructor_alternative(self):

        """This function is the complement to
        'function_constructor_alternative', but instead constructs the
        derivative of the map's component functions.

        It constructs the functions' strings, then converts them into
        functions.
        """

        self.der_fun_mon = []
        self.der_fun_mon_strings = []

        self.optimization_constraints_lb = []
        self.optimization_constraints_ub = []

        # Find out how many function terms we are building
        K = len(self.monotone)

        # Go through all terms
        for k in range(K):

            # Step 1: Build the monotone function

            # Set optimization constraints
            self.optimization_constraints_lb.append(np.zeros(len(self.monotone[k])))
            self.optimization_constraints_ub.append(
                np.ones(len(self.monotone[k])) * np.inf
            )

            # Define modules to load
            modules = [" ", ""]

            # Define the terms composing the transport map component
            terms = []

            # Prepare a counter for the special terms
            ST_counter = np.zeros(self.X.shape[-1], dtype=int)

            # Mark which of these are special terms, in case we want to create
            # permutations of multiple RBFS
            ST_indices = []

            # Go through all terms, extract terms for precalculation
            dict_precalc = {}
            for j, entry in enumerate(self.monotone[k]):

                # Convert the map specification to a function

                # Find the term's function
                term, modifier_log = self.write_basis_function(
                    term=entry, mode="derivative", k=k + self.skip_dimensions
                )

                # If this is a constant term, undo the lower constraint

                if "constant" in list(modifier_log.keys()):
                    # Assign linear constraints
                    self.optimization_constraints_lb[k][j] = -np.inf
                    self.optimization_constraints_ub[k][j] = +np.inf

                    # Extract any precalculations, where applicable

                # If this term includes and precalculations, extract them
                if "variables" in list(modifier_log.keys()):

                    # There are precalculating variables. Go through each
                    for key in list(modifier_log["variables"].keys()):

                        # Have we logged this one already?
                        if key not in list(dict_precalc.keys()):
                            # No, we haven't. Add it.
                            dict_precalc[key] = copy.copy(
                                modifier_log["variables"][key]
                            ).replace("__x__", "x")

                            # Post-processing for special terms

                # Is this term a special term?
                if "ST" in list(modifier_log.keys()):

                    # Mark this term as a special one
                    ST_indices.append(j)

                    # Yes, it is. Add additional modules to load, if necessary
                    if "import scipy.special" not in modules:
                        modules.append("import scipy.special")

                    # Extract this special term's dimension
                    idx = modifier_log["ST"]

                    # Replace __mu__ with the correct ST location variable
                    term = term.replace(
                        "__mu__",
                        "self.ST_centers_m["
                        + str(k)
                        + "]["
                        + str(idx)
                        + "]["
                        + str(ST_counter[idx])
                        + "]",
                    )

                    # Replace __scale__ with the correct ST location variable
                    term = term.replace(
                        "__scale__",
                        "self.ST_scales_m["
                        + str(k)
                        + "]["
                        + str(idx)
                        + "]["
                        + str(ST_counter[idx])
                        + "]",
                    )

                    # Increment the special term counter
                    ST_counter[idx] += 1

                    # Add the term to the list

                # If any dummy __x__ remain, replace them
                term = term.replace("__x__", "x")

                # Store the term
                terms.append(copy.copy(term))

            # Are there multiple special terms?
            if (
                np.sum([True if x != 0 else False for x in self.RBF_counter_m[k, :]])
                > 1
            ):

                # Yes, there are multiple special terms. Extract these terms.
                RBF_terms = [terms[i] for i in ST_indices]

                # Check what variables these terms are affiliated with
                RBF_terms_dim = -np.ones(len(RBF_terms), dtype=int)
                for ki in range(k + 1 + self.skip_dimensions):
                    for i, term in enumerate(RBF_terms):
                        if "x[...," + str(ki) + "]" in term:
                            RBF_terms_dim[i] = ki
                RBF_terms_dims = np.unique(np.asarray(RBF_terms_dim))

                # Create a dictionary with the different terms
                RBF_terms_dict = {}
                for i in RBF_terms_dims:
                    RBF_terms_dict[i] = [
                        RBF_terms[j]
                        for j in range(len(RBF_terms))
                        if RBF_terms_dim[j] == i
                    ]

                # Create all combinations of terms
                RBF_terms_grid = copy.deepcopy(RBF_terms_dict[RBF_terms_dims[0]])
                for i in RBF_terms_dims[1:]:
                    # Create a grid with the next dimension
                    RBF_terms_grid = list(
                        itertools.product(
                            RBF_terms_grid, copy.deepcopy(RBF_terms_dict[i])
                        )
                    )

                    # Convert this list of tuples into a new list of strings
                    RBF_terms_grid = [
                        entry[0] + "*" + entry[1] for entry in RBF_terms_grid
                    ]

                # Now remove all original RBF terms
                terms = [entry for i, entry in enumerate(terms) if i not in ST_indices]

                # Now add all the grid terms
                terms += RBF_terms_grid

            # Assemble the monotone derivative function

            # Prepare the basis string
            string = "def fun(x,self):\n\t\n\t"

            # Load module requirements

            # Add all module requirements
            for entry in modules:
                string += copy.copy(entry) + "\n\t"
            string += "\n\t"  # Another line break for legibility

            # Prepare linearization, if necessary

            # If linearization is active, truncate the input x
            if self.linearization is not None:
                # First, find our which parts are outside the linearization hypercube
                string += "vec_below = self.linearization_threshold[:,0][np.newaxis,:] - x;\n\t"
                string += (
                    "vec_below[vec_below > 0] = 0;\n\t"  # Set all values above to zero
                )
                string += "vec_above = x - self.linearization_threshold[:,1][np.newaxis,:];\n\t"
                string += (
                    "vec_above[vec_above > 0] = 0;\n\t"  # Set all values below to zero
                )
                string += "vec = vec_above + vec_below;\n\t"

                # Then convert the two arrays to boolean markers
                string += "below = (vec_below < 0)\n\t"
                string += "above = (vec_above > 0);\n\t"
                string += "vecnorm = np.asarray(below,dtype=float) + np.asarray(above,dtype=float);\n\t"

                # Truncate all values outside the hypercube
                string += "for d in range(x.shape[1]):\n\t\t"
                string += "x[below[:,d],d] = self.linearization_threshold[d,0];\n\t\t"
                string += "x[above[:,d],d] = self.linearization_threshold[d,1];\n\t"

                # Add a space to the next block
                string += "\n\t"

                # The derivative of a linearized function outside its range is
                # constant, so we do not require x_ext

                # Prepare precalculated variables

            # Add all precalculation terms
            for key in list(dict_precalc.keys()):
                string += key + " = " + copy.copy(dict_precalc[key]) + ";\n\t"

                # Assemble function output

            # Prepare the result string
            if len(terms) == 1:  # Only a single term, no need for stacking

                string += "result = " + copy.copy(terms[0]) + "[:,np.newaxis];\n\t\n\t"

            else:  # If we have more than one term, start stacking the result

                # Prepare the stack
                string += "result = np.stack((\n\t\t"

                # Go through each entry in terms, add them one by one
                for entry in terms:
                    string += copy.copy(entry) + ",\n\t\t"

                # Remove the last ",\n\t\t" and close the stack
                string = string[:-4]
                string += "),axis=-1)\n\t\n\t"

            # Return the result
            string += "return result"

            # Finish function construction

            # Append the function string
            self.der_fun_mon_strings.append(string)

            # Create an actual function
            funstring = "der_fun_mon_" + str(k)
            exec(string.replace("fun", funstring), globals())
            exec("self.der_fun_mon.append(copy.deepcopy(" + funstring + "))")

    def check_for_special_terms(self):

        """This function scans through the user-provided map specifications and
        seeks if there are any special terms ('RBF', 'iRBF', 'LET', 'RET')
        among the terms of the map components.

        If there are, it determines how many there are, and informs the
        rest of the function where these special terms should be
        located.
        """

        # Number of RBFs
        self.special_terms = []

        # Create a matrix of length-of-map - by - number-of-X-dimensions to
        # store how many RBFs, IntRBFs or ETs are defined
        self.RBF_counter_m = np.zeros((len(self.monotone), self.X.shape[1]), dtype=int)
        self.RBF_counter_nm = np.zeros(
            (len(self.nonmonotone), self.X.shape[1]), dtype=int
        )

        # Create a list keeping track of what dimensions the special terms
        # occupy; we use this to create grids of multidimensional RBFs
        self.RBF_dims_nm = []
        self.RBF_dims_m = []

        # Go through all map components
        for k in range(self.D):

            # Append an empty list
            self.RBF_dims_nm.append([])
            self.RBF_dims_m.append([])

            # Check all non-monotone terms of this map component
            for entry in self.nonmonotone[k]:

                # If this term is a string, it denotes a special term
                if type(entry) == str:
                    # Split the entry and extract its dimensional entry
                    index = int(entry.split(" ")[1])

                    # Mark it in memory
                    self.RBF_counter_nm[k, index] += 1

            # Check all monotone terms of this map component
            for entry in self.monotone[k]:

                # If this term is a string, it denotes a special term
                if type(entry) == str:
                    # Split the entry and extract its dimensional entry
                    index = int(entry.split(" ")[1])

                    # Mark it in memory
                    self.RBF_counter_m[k, index] += 1

    def calculate_special_term_locations(self):

        """This function calculates the location and scale parameters for
        special terms in the transport map definition, specifically RBF (Radial
        Basis Functions), iRBF (Integrated Radial Basis Functions), and LET/RET
        (Edge Terms).

        Position and scale parameters are assigned in the order they
        have been defined, so make sure to define a left edge term first
        if you want it to be on the left side.
        """

        self.ST_centers_m = []  # Centers of RBFs in monotone terms
        self.ST_scales_m = []  # Scales of RBFs in monotone terms

        self.ST_centers_nm = []  # Centers of RBFs in non-monotone terms
        self.ST_scales_nm = []  # Scales of RBFs in non-monotone terms

        self.linearization_threshold = np.zeros((self.X.shape[-1], 2))

        for k in range(self.D):

            for idx, RBF_counter in enumerate(
                [self.RBF_counter_m, self.RBF_counter_nm]
            ):

                if np.sum(RBF_counter[k, :]) == 0:  # No special terms are defined

                    if idx == 0:  # monotone

                        self.ST_centers_m.append(None)
                        self.ST_scales_m.append(None)

                    elif idx == 1:  # non-monotone

                        self.ST_centers_nm.append(None)
                        self.ST_scales_nm.append(None)

                else:  # Some special terms are defined

                    if idx == 0:  # monotone

                        self.ST_centers_m.append([])
                        self.ST_scales_m.append([])

                    elif idx == 1:  # non-monotone

                        self.ST_centers_nm.append([])
                        self.ST_scales_nm.append([])

                    # Go through all dimensions
                    for d in range(self.X.shape[-1]):

                        # Zero special terms

                        if RBF_counter[k, d] == 0:  # No ST for this dimension

                            if idx == 0:  # monotone

                                self.ST_centers_m[-1].append([])
                                self.ST_scales_m[-1].append([])

                            elif idx == 1:  # non-monotone

                                self.ST_centers_nm[-1].append([])
                                self.ST_scales_nm[-1].append([])

                                # One special term

                        elif RBF_counter[k, d] == 1:  # One ST is special

                            if idx == 0:  # monotone

                                self.ST_centers_m[-1].append(
                                    [np.quantile(self.X[:, d], q=0.5)]
                                )

                                if self.ST_scale_mode == "dynamic":
                                    self.ST_scales_m[-1].append(
                                        [self.ST_scale_factor / 2]
                                    )
                                elif self.ST_scale_mode == "static":
                                    self.ST_scales_m[-1].append([self.ST_scale_factor])

                            elif idx == 1:  # non-monotone

                                self.ST_centers_nm[-1].append(
                                    [np.quantile(self.X[:, d], q=0.5)]
                                )

                                # self.ST_scales_nm[-1]   .append([self.ST_scale_factor/2])

                                if self.ST_scale_mode == "dynamic":
                                    self.ST_scales_nm[-1].append(
                                        [self.ST_scale_factor / 2]
                                    )
                                elif self.ST_scale_mode == "static":
                                    self.ST_scales_nm[-1].append([self.ST_scale_factor])

                                    # Multiple special terms

                        else:  # We have two or more STs

                            # Decide where to place the special terms
                            quantiles = np.arange(1, RBF_counter[k, d] + 1, 1) / (
                                RBF_counter[k, d] + 1
                            )

                            # Append an empty list, then fill it
                            scales = np.zeros(RBF_counter[k, d])

                            if idx == 0:  # monotone

                                self.ST_centers_m[-1].append(
                                    np.quantile(a=self.X[:, d], q=quantiles)
                                )

                                if self.ST_scale_mode == "dynamic":

                                    # Otherwise, determine the scale based on relative differences
                                    for i in range(RBF_counter[k, d]):

                                        # Left edge-case: base is half distance to next basis
                                        if i == 0:

                                            scales[i] = (
                                                (
                                                    self.ST_centers_m[-1][d][1]
                                                    - self.ST_centers_m[-1][d][0]
                                                )
                                                / 2
                                                * self.ST_scale_factor
                                            )

                                        # Right edge-case: base is half distance to previous basis
                                        elif i == RBF_counter[k, d] - 1:

                                            scales[i] = (
                                                (
                                                    self.ST_centers_m[-1][d][i]
                                                    - self.ST_centers_m[-1][d][i - 1]
                                                )
                                                / 2
                                                * self.ST_scale_factor
                                            )

                                        # Otherwise: base is average distance to neighbours
                                        else:

                                            scales[i] = (
                                                (
                                                    self.ST_centers_m[-1][d][i + 1]
                                                    - self.ST_centers_m[-1][d][i - 1]
                                                )
                                                / 2
                                                * self.ST_scale_factor
                                            )

                                    # Copy the scales into the array
                                    self.ST_scales_m[-1].append(copy.copy(scales))

                                elif self.ST_scale_mode == "static":

                                    self.ST_scales_m[-1].append(
                                        copy.copy(scales) + self.ST_scale_factor
                                    )

                            elif idx == 1:  # non-monotone

                                self.ST_centers_nm[-1].append(
                                    np.quantile(a=self.X[:, d], q=quantiles)
                                )

                                if self.ST_scale_mode == "dynamic":

                                    # Otherwise, determine the scale based on relative differences
                                    for i in range(RBF_counter[k, d]):

                                        # Left edge-case: base is half distance to next basis
                                        if i == 0:

                                            scales[i] = (
                                                (
                                                    self.ST_centers_nm[-1][d][1]
                                                    - self.ST_centers_nm[-1][d][0]
                                                )
                                                / 2
                                                * self.ST_scale_factor
                                            )

                                        # Right edge-case: base is half distance to previous basis
                                        elif i == RBF_counter[k, d] - 1:

                                            scales[i] = (
                                                (
                                                    self.ST_centers_nm[-1][d][i]
                                                    - self.ST_centers_nm[-1][d][i - 1]
                                                )
                                                / 2
                                                * self.ST_scale_factor
                                            )

                                        # Otherwise: base is average distance to neighbours
                                        else:

                                            scales[i] = (
                                                (
                                                    self.ST_centers_nm[-1][d][i + 1]
                                                    - self.ST_centers_nm[-1][d][i - 1]
                                                )
                                                / 2
                                                * self.ST_scale_factor
                                            )

                                    # Copy the scales into the array
                                    self.ST_scales_nm[-1].append(copy.copy(scales))

                                elif self.ST_scale_mode == "static":

                                    self.ST_scales_nm[-1].append(
                                        copy.copy(scales) + self.ST_scale_factor
                                    )

        # Set the linearization bounds
        if self.linearization is not None:

            # If the linearization value specifies quantiles, calculate the linearization thresholds
            if self.linearization_specified_as_quantiles:

                for k in range(self.X.shape[-1]):
                    self.linearization_threshold[k, 0] = np.quantile(
                        self.X[:, k], q=self.linearization
                    )
                    self.linearization_threshold[k, 1] = np.quantile(
                        self.X[:, k], q=1 - self.linearization
                    )

            # Otherwise, directly prescribe them
            else:

                for k in range(self.X.shape[-1]):
                    # Overwrite with static term -marked-
                    self.linearization_threshold[k, 0] = -self.linearization
                    self.linearization_threshold[k, 1] = +self.linearization

        return

    def map(self, X: np.array = None):

        """This function maps the samples X from the target distribution to the
        standard multivariate Gaussian reference distribution. If X has not
        been provided, the samples in storage will be used instead.

        :param X: N-by-D array of the training samples used to optimize the transport map, where N is the number of
            samples and D is the number of dimensions.
        :return: N-by-D array of the mapped samples
        """

        if X is not None and self.standardize_samples:
            X = self.scaler.fit_transform(X)
        elif X is None:
            X = self.X
        # Initialize the output array
        Y = np.zeros((X.shape[0], self.D))

        for k in range(self.D):
            # Apply the forward map
            Y[:, k] = self.s(
                x=X,
                k=k,
                coeffs_nonmon=self.coeffs_nonmon[k],
                coeffs_mon=self.coeffs_mon[k],
            )

        return Y

    def s(
        self,
        x: np.array,
        k: int,
        coeffs_nonmon: np.array = None,
        coeffs_mon: np.array = None,
    ):

        """This function evaluates the k-th map component.

        :param x: N-by-D array of the training samples used to optimize the transport map, where N is the number of
            samples and D is the number of dimensions. Can be None, at which point it is replaced with X from storage.
        :param k: an integer variable defining what map component is being evaluated. Corresponds to a dimension of
            sample space.
        :param coeffs_nonmon: a vector specifying the coefficients of the non-monotone part of the map component's
            terms, i.e., those entries which do not depend on x_k. This vector is replaced from storage if it is not
            overwritten.
        :param coeffs_mon: a vector specifying the coefficients of the monotone part of the map component's terms, i.e.,
            those entries which do not depend on x_k. This vector is replaced from storage if it is not overwritten.
        :return: N-by-1 array of the k-th map component evaluated at the samples in x
        """

        # Load in values if required
        if x is None:
            x = self.X
            Psi_nonmon = copy.copy(self.Psi_nonmon[k])
        else:
            Psi_nonmon = copy.copy(self.fun_nonmon[k](x, self))

        if coeffs_mon is None:
            coeffs_mon = self.coeffs_mon[k]

        if coeffs_nonmon is None:
            coeffs_nonmon = self.coeffs_nonmon[k]

            # Calculate the non-monotone part

        if Psi_nonmon is not None:
            nonmonotone_part = np.dot(Psi_nonmon, coeffs_nonmon[:, np.newaxis])[..., 0]
        else:
            nonmonotone_part = 0

            # Calculate the monotone part

        if self.monotonicity == "integrated rectifier":

            # Prepare the integration argument
            def integral_argument(x_, y, coeffs_mon_, k_):

                # First reconstruct the full X matrix
                X_loc = y
                X_loc[:, self.skip_dimensions + k_] = x_

                # Then evaluate the Psi matrix
                Psi_mon_loc = self.fun_mon[k_](X_loc, self)

                # Determine the gradients
                rect_arg = np.dot(Psi_mon_loc, coeffs_mon_[:, np.newaxis])[..., 0]

                # Send the rectifier argument through the rectifier
                arg = self.rect.evaluate(rect_arg)

                # If there is any delta term to prevent underflow, add it
                arg += self.delta

                return arg

            # Evaluate the integral
            monotone_part = self.gauss_quadrature(
                f=integral_argument,
                a=0,
                b=x[..., self.skip_dimensions + k],
                args=(x, coeffs_mon, k),
                **self.quadrature_input,
            )

        elif self.monotonicity == "separable monotonicity":

            # In the case that monotonicity is enforced through parameterization,
            # simply evaluate the monotone function
            monotone_part = np.dot(self.fun_mon[k](x, self), coeffs_mon[:, np.newaxis])[
                :, 0
            ]

        # Combine the terms
        result = copy.copy(nonmonotone_part + monotone_part)

        return result

    def optimize(self):

        """This function optimizes the map's component functions, seeking the
        coefficients which best map the samples to a standard multivariate
        Gaussian distribution."""

        if self.workers == 1:  # With only one worker, don't parallelize

            # The standard optimization pathway, most flexibility
            if self.monotonicity == "integrated rectifier":

                # Go through all map components
                for k in range(self.D):

                    # Optimize this map component
                    results = self.worker_task(k=k, task_supervisor=None)

                    # Print optimization progress
                    if self.verbose:
                        string = "\r" + "Progress: |"
                        string += (k + 1) * "█"
                        string += (self.D - k - 1) * " "
                        string += "|"

                        logger.info(string, end="\r")

                    # Extract and store the optimized coefficients
                    self.coeffs_nonmon[k] = copy.deepcopy(results[0])
                    self.coeffs_mon[k] = copy.deepcopy(results[1])

            # A faster, albeit less flexible optimization pathway
            elif self.monotonicity == "separable monotonicity":

                # Go through all map components
                for k in range(self.D):

                    # Optimize this map component
                    results = self.worker_task_monotone(k=k, task_supervisor=None)

                    # Print optimization progress
                    if self.verbose:
                        string = "\r" + "Progress: |"
                        string += (k + 1) * "█"
                        string += (self.D - k - 1) * " "
                        string += "|"
                        logger.info(string, end="\r")

                    # Extract and store the optimized coefficients
                    self.coeffs_nonmon[k] = results[0]
                    self.coeffs_mon[k] = results[1]

        elif self.workers > 1:  # If we have more than one worker, parallelize

            from multiprocessing import Pool, Manager
            from itertools import repeat

            # Prepare parallelization

            # Create the task supervisor
            manager = Manager()
            task_supervisor = manager.list([0] * self.D)
            p = Pool(processes=self.workers)

            # Start parallel tasks

            if self.monotonicity == "integrated rectifier":

                # For parallelization, Python seemingly cannot share functions we have
                # dynamically assembled between processes. As a consequence, we must
                # delete them, then re-create them inside the processes
                del self.fun_mon, self.fun_nonmon

                # Start the worker
                # We flip the order of the tasks because components farther down in the
                # transport map take longer to computer; it is computationally useful
                # to tackle these tasks first, so we don't leave the longest task last
                results = p.starmap(
                    func=self.worker_task,
                    iterable=zip(np.flip(np.arange(self.D)), repeat(task_supervisor)),
                )
                p.join()

            elif self.monotonicity == "separable monotonicity":

                # For parallelization, Python seemingly cannot share functions we have
                # dynamically assembled between processes. As a consequence, we must
                # delete them, then re-create them inside the processes
                del self.fun_mon, self.fun_nonmon, self.der_fun_mon

                # Start the worker
                # We flip the order of the tasks because components farther down in the
                # transport map take longer to computer; it is computationally useful
                # to tackle these tasks first, so we don't leave the longest task last
                results = p.starmap(
                    func=self.worker_task_monotone,
                    iterable=zip(np.flip(np.arange(self.D)), repeat(task_supervisor)),
                )
                p.join()

                # Post-process parallel task

            if self.verbose:
                # Make final update to the task supervisor
                string = "\r" + "Progress: |"
                for i in range(len(task_supervisor)):
                    if task_supervisor[i] == 1:  # Successful task
                        string += "█"
                    elif task_supervisor[i] == -1:  # (Partially) failed task
                        string += "X"
                    elif task_supervisor[i] == 2:  # Successful task upon restart
                        string += "R"
                    else:
                        string += " "  # Unfinished task (should not occur)
                string += "|"
                logger.info(string)

            # Reverse the results back into proper order
            results.reverse()

            # Go through all results and save the coefficients
            for k in range(self.D):
                # Save the coefficients
                self.coeffs_nonmon[k] = copy.deepcopy(results[k][0])
                self.coeffs_mon[k] = copy.deepcopy(results[k][1])

            # Restore the functions we previously deleted
            self.fun_mon = []
            self.fun_nonmon = []
            if self.monotonicity == "separable monotonicity":
                self.der_fun_mon = []
            for k in range(self.D):

                # Create the function
                funstring = "fun_mon_" + str(k)
                exec(self.fun_mon_strings[k].replace("fun", funstring), globals())
                exec("self.fun_mon.append(copy.deepcopy(" + funstring + "))")

                # Create the function
                funstring = "fun_nonmon_" + str(k)
                exec(self.fun_nonmon_strings[k].replace("fun", funstring), globals())
                exec("self.fun_nonmon.append(copy.deepcopy(" + funstring + "))")

                if self.monotonicity == "separable monotonicity":
                    # Create the function
                    funstring = "der_fun_mon_" + str(k)
                    exec(
                        self.der_fun_mon_strings[k].replace("fun", funstring), globals()
                    )
                    exec("self.der_fun_mon.append(copy.deepcopy(" + funstring + "))")
            p.close()

    def worker_task_monotone(self, k: int, task_supervisor: list):

        """This function provides the optimization task for the k-th map component function to a worker
        (if parallelization is used), or applies it in sequence (if no parallelization is used).
        This specific function only becomes active if monotonicity = 'separable monotonicity'.

        :param k: an integer variable defining what map component is being evaluated. Corresponds to a dimension of
            sample space.
        :param task_supervisor: a shared list which informs the main process how many optimization tasks have already
            been computed. This list should not be specified by the user, it only serves to provide information about
            the optimization progress.
        :return: a list containing the optimized coefficients for the k-th map component function.
        """

        # Prepare task

        if task_supervisor is not None and self.verbose:

            # Print multiprocessing progress
            string = "\r" + "Progress: |"
            for i in range(len(task_supervisor)):
                if task_supervisor[i] == 1:
                    string += "█"
                elif task_supervisor[i] == -1:
                    string += "X"
                elif task_supervisor[i] == 2:
                    string += "R"
                else:
                    string += " "
            string += "|"
            logger.info(string, end="\r")

        coeffs_nonmon = self.coeffs_nonmon[k]
        coeffs_mon = self.coeffs_mon[k]

        # Re-create the functions

        # Restore the functions we previously deleted
        self.der_fun_mon = []
        self.fun_mon = []
        self.fun_nonmon = []
        for i in range(self.D):
            # Create the derivative of the monotone function
            funstring = f"der_fun_mon_{i}"
            exec(self.der_fun_mon_strings[i].replace("fun", funstring), globals())
            exec(f"self.der_fun_mon.append(copy.deepcopy({funstring}))")

            # Create the monotone function
            funstring = f"fun_mon_{i}"
            exec(self.fun_mon_strings[i].replace("fun", funstring), globals())
            exec(f"self.fun_mon.append(copy.deepcopy({funstring}))")

            # Create the non-monotone function
            funstring = f"fun_nonmon_{i}"
            exec(self.fun_nonmon_strings[i].replace("fun", funstring), globals())
            exec(f"self.fun_nonmon.append(copy.deepcopy({funstring}))")

            # Define special objective for the monotone function

        Psi_mon = copy.copy(self.fun_mon[k](copy.copy(self.X), self))

        Psi_nonmon = copy.copy(self.fun_nonmon[k](copy.copy(self.X), self))

        if self.regularization is None:

            # No regularization, use standard objective

            Q, R = np.linalg.qr(Psi_nonmon, mode="reduced")

            A_sqrt = Psi_mon - np.linalg.multi_dot((Q, Q.T, Psi_mon))

            N = self.X.shape[0]  # Ensemble size
            A = np.dot(A_sqrt.T, A_sqrt) / N

            def fun_mon_objective(coeffs_mon_, A_, k_):

                # First part: How close is the ensemble mapped to zero?

                objective = (
                    np.linalg.multi_dot(
                        (coeffs_mon_[:, np.newaxis].T, A_, coeffs_mon_[:, np.newaxis])
                    )[0, 0]
                    / 2
                )

                # Second part: How much is the ensemble inflated?

                der_Psi_mon = copy.copy(self.der_fun_mon[k_](copy.copy(self.X), self))

                # Determine the gradients of the polynomial functions
                monotone_part_der = np.dot(der_Psi_mon, coeffs_mon_[:, np.newaxis])[
                    ..., 0
                ]

                # Subtract this from the objective
                objective -= np.mean(np.log(monotone_part_der))

                return objective

        elif self.regularization.lower() == "l2":

            # L2 regularization, use alternative objective

            # Step 1: Calculate basis for the supporting variable A
            A = np.linalg.multi_dot(
                (
                    np.linalg.inv(
                        np.dot(Psi_nonmon.T, Psi_nonmon)
                        + self.regularization_lambda * np.identity(Psi_nonmon.shape[-1])
                    ),
                    Psi_nonmon.T,
                    Psi_mon,
                )
            )

            # Step 2: Aggregate
            A = np.dot(
                (Psi_mon - np.dot(Psi_nonmon, A)).T, Psi_mon - np.dot(Psi_nonmon, A)
            ) / 2 + self.regularization_lambda * (
                np.dot(A.T, A) + np.identity(A.shape[-1])
            )

            # Create the objective function
            def fun_mon_objective(coeffs_mon_, A_, k_):

                # First part: How close is the ensemble mapped to zero?

                objective = np.linalg.multi_dot(
                    (coeffs_mon_[:, np.newaxis].T, A_, coeffs_mon_[:, np.newaxis])
                )[0, 0]

                # Second part: How much is the ensemble inflated?

                der_Psi_mon = copy.copy(self.der_fun_mon[k_](copy.copy(self.X), self))

                # Determine the gradients of the polynomial functions
                monotone_part_der = np.dot(der_Psi_mon, coeffs_mon_[:, np.newaxis])[
                    ..., 0
                ]

                # Subtract this from the objective
                objective -= np.sum(np.log(monotone_part_der))

                return objective

                # Call the optimization routine

        bounds = []
        for idx in range(len(self.optimization_constraints_lb[k])):
            bounds.append(
                [
                    self.optimization_constraints_lb[k][
                        idx
                    ],  # -marked- used to have +1E-8
                    self.optimization_constraints_ub[k][idx],
                ]
            )

        opt = minimize(
            fun=fun_mon_objective,
            method="L-BFGS-B",
            x0=coeffs_mon,
            bounds=bounds,
            args=(A, k),
        )

        # Post-process the optimization results

        # Retrieve the optimized coefficients
        coeffs_mon = copy.copy(opt.x)

        if task_supervisor is not None and self.verbose:

            # If optimization was a success, mark it as such
            if opt.success:

                # '1' represents initial success ('█')
                task_supervisor[k] = 1

            else:

                # '-1' represents failure ('X')
                task_supervisor[k] = -1

            # Print multiprocessing progress
            string = "\r" + "Progress: |"
            for i in range(len(task_supervisor)):
                if task_supervisor[i] == 1:
                    string += "█"
                elif task_supervisor[i] == -1:
                    string += "X"
                elif task_supervisor[i] == 2:
                    string += "R"
                else:
                    string += " "
            string += "|"
            logger.info(string, end="\r")

            # With the monotone coefficients found, calculate the non-monotone coeffs

        if self.regularization is None:

            # In the standard formulation, use the QR decomposition to calculate
            # the non-monotone coefficients
            coeffs_nonmon = -np.linalg.multi_dot(
                (np.linalg.inv(R), Q.T, Psi_mon, coeffs_mon[:, np.newaxis])
            )[:, 0]

        elif self.regularization.lower() == "l2":

            coeffs_nonmon = -np.linalg.multi_dot(
                (
                    np.linalg.inv(
                        np.dot(self.Psi_nonmon[k].T, self.Psi_nonmon[k])
                        + 2
                        * self.regularization_lambda
                        * np.identity(self.Psi_nonmon[k].shape[-1])
                    ),
                    np.dot(self.Psi_nonmon[k].T, self.Psi_mon[k]),
                    coeffs_mon[:, np.newaxis],
                )
            )[:, 0]

        # Return both optimized coefficients
        return coeffs_nonmon, coeffs_mon

    def worker_task(self, k: int, task_supervisor: list):

        """This function provides the optimization task for the k-th map component function to a worker
        (if parallelization is used), or applies it in sequence (if no parallelization is used).
        This specific function only becomes active if monotonicity = 'integrated rectifier'.

        :param k: an integer variable defining what map component is being evaluated. Corresponds to a dimension of
            sample space.
        :param task_supervisor: a shared list which informs the main process how many optimization tasks have already
            been computed. This list should not be specified by the user, it only serves to provide information about the
             optimization progress.
        """

        # Prepare task

        if task_supervisor is not None and self.verbose:

            # Print multiprocessing progress
            string = "\r" + "Progress: |"
            for i in range(len(task_supervisor)):
                if task_supervisor[i] == 1:
                    string += "█"
                elif task_supervisor[i] == -1:
                    string += "X"
                elif task_supervisor[i] == 2:
                    string += "R"
                else:
                    string += " "
            string += "|"
            logger.info(string, end="\r")

        # Assemble the theta vector we are optimizing
        coeffs = np.zeros(len(self.coeffs_nonmon[k]) + len(self.coeffs_mon[k]))
        div = len(self.coeffs_nonmon[k])  # Divisor for the vector

        # Write in the coefficients
        coeffs[:div] = copy.copy(self.coeffs_nonmon[k])
        coeffs[div:] = copy.copy(self.coeffs_mon[k])

        # Re-create the functions

        if self.workers > 1:

            # Restore the functions we previously deleted
            self.fun_mon = []
            self.fun_nonmon = []
            for i in range(self.D):
                # Create the function
                funstring = f"fun_mon_{i}"
                exec(self.fun_mon_strings[i].replace("fun", funstring), globals())
                exec(f"self.fun_mon.append(copy.deepcopy({funstring}))")

                # Create the function
                funstring = f"fun_nonmon_{i}"
                exec(self.fun_nonmon_strings[i].replace("fun", funstring), globals())
                exec(f"self.fun_nonmon.append(copy.deepcopy({funstring}))")

        # Call the optimization routine

        # Minimize the objective function
        opt = minimize(
            method="BFGS",  # 'L-BFGS-B',
            fun=self.objective_function,
            jac=self.objective_function_jacobian,
            x0=coeffs,
            args=(k, div),
        )

        # Post-process the optimization results

        # Retrieve the optimized coefficients
        coeffs_opt = copy.copy(opt.x)

        # Separate them into coefficients for monotone and non-monotone parts
        coeffs_nonmon = coeffs_opt[:div]
        coeffs_mon = coeffs_opt[div:]

        if task_supervisor is not None and self.verbose:

            # If optimization was a success, mark it as such
            if opt.success:

                # '1' represents initial success ('█')
                task_supervisor[k] = 1

            else:

                # '-1' represents failure ('X')
                task_supervisor[k] = -1

            # Print multiprocessing progress
            string = "\r" + "Progress: |"
            for i in range(len(task_supervisor)):
                if task_supervisor[i] == 1:
                    string += "█"
                elif task_supervisor[i] == -1:
                    string += "X"
                elif task_supervisor[i] == 2:
                    string += "R"
                else:
                    string += " "
            string += "|"
            logger.info(string, end="\r")

        # Return both optimized coefficients
        return coeffs_nonmon, coeffs_mon

    def objective_function(self, coeffs: np.array, k: int, div: int = 0):

        """This function evaluates the objective function used in the
        optimization of the map's component functions.

        :param coeffs: a vector containing the coefficients for both the non-monotone and monotone terms of the k-th map
            component function. Is replaced for storage is specified as None.
        :param k: an integer variable defining what map component
        :param div: an integer specifying where the cutoff between the non-monotone and monotone coefficients in
            'coeffs' is.
        :return: the value of the objective function
        """

        # Partition the coefficient vector, if necessary
        if coeffs is not None:
            # Separate the vector into non-monotone and monotone coefficients
            coeffs_nonmon = copy.copy(coeffs[:div])
            coeffs_mon = copy.copy(coeffs[div:])
        else:
            if self.verbose:
                logger.info("loading")
            # Otherwise, load them from object
            coeffs_nonmon = copy.copy(self.coeffs_nonmon[k])
            coeffs_mon = copy.copy(self.coeffs_mon[k])

            # First part: How close is the ensemble mapped to zero?

        # Map the samples to the reference marginal
        map_result = self.s(
            x=None, k=k, coeffs_nonmon=coeffs_nonmon, coeffs_mon=coeffs_mon
        )

        # Check how close these samples are to the origin
        objective = 1 / 2 * map_result ** 2

        # Second part: How much is the ensemble inflated?

        Psi_mon = self.fun_mon[k](self.X, self)

        # Determine the gradients of the polynomial functions
        monotone_part_der = np.dot(Psi_mon, coeffs_mon[:, np.newaxis])[..., 0]

        # Evaluate the logarithm of the rectified monotone part
        obj = self.rect.logevaluate(monotone_part_der)

        # Subtract this from the objective
        objective -= obj

        # Average the objective function

        # Now summarize the contributions and take their average
        objective = np.mean(objective)

        # Add regularization, if desired

        if self.regularization is not None:

            # A scalar regularization was specified
            if type(self.regularization) == str:

                if self.regularization.lower() == "l1":

                    # Regularization_lambda is identical for all parameters
                    if np.isscalar(self.regularization_lambda):

                        # Add l1 regularization for all coefficients
                        objective += self.regularization_lambda * np.sum(
                            np.abs(coeffs_mon)
                        )
                        objective += self.regularization_lambda * np.sum(
                            np.abs(coeffs_nonmon)
                        )

                    elif type(self.regularization_lambda) == list:

                        # Add l1 regularization for all coefficients
                        objective += np.sum(
                            self.regularization_lambda[k][div:] * np.abs(coeffs_mon)
                        )
                        objective += np.sum(
                            self.regularization_lambda[k][:div] * np.abs(coeffs_nonmon)
                        )

                    else:

                        raise ValueError(
                            "Data type of regularization_lambda not understood. Must be either scalar or list."
                        )

                elif self.regularization.lower() == "l2":

                    # Regularization_lambda is identical for all parameters
                    if np.isscalar(self.regularization_lambda):

                        # Add l2 regularization for all coefficients
                        objective += self.regularization_lambda * np.sum(
                            coeffs_mon ** 2
                        )
                        objective += self.regularization_lambda * np.sum(
                            coeffs_nonmon ** 2
                        )

                    elif type(self.regularization_lambda) == list:

                        # Add l1 regularization for all coefficients
                        objective += np.sum(
                            self.regularization_lambda[k][div:] * coeffs_mon ** 2
                        )
                        objective += np.sum(
                            self.regularization_lambda[k][:div] * coeffs_nonmon ** 2
                        )

                    else:

                        raise ValueError(
                            "Data type of regularization_lambda not understood. Must be either scalar or list."
                        )

                else:

                    raise ValueError("regularization_type must be either 'l1' or 'l2'.")

            else:

                raise ValueError(
                    "The variable 'regularization' must be either None, 'l1', or 'l2'."
                )

        return objective

    def objective_function_jacobian(self, coeffs: np.array, k: int, div: int = 0):

        """This function evaluates the derivative of the objective function
        used in the optimization of the map's component functions.

        :param coeffs: a vector containing the coefficients for both the non-monotone and monotone terms of the k-th map
             component function. Is replaced for storage is specified as None.
        :param k: an integer variable defining what map component is being evaluated. Corresponds to a dimension of
            sample space.
        :param div: an integer specifying where the cutoff between the non-monotone and monotone coefficients in
            'coeffs' is.
        """

        # Partition the coefficient vector, if necessary
        if coeffs is not None:
            # Separate the vector into non-monotone and monotone coefficients
            coeffs_nonmon = copy.copy(coeffs[:div])
            coeffs_mon = copy.copy(coeffs[div:])
        else:
            # Otherwise, load them from object
            coeffs_nonmon = copy.copy(self.coeffs_nonmon[k])
            coeffs_mon = copy.copy(self.coeffs_mon[k])

        # Prepare term 1

        # First, handle the scalar
        term_1_scalar = self.s(
            x=None, k=k, coeffs_nonmon=coeffs_nonmon, coeffs_mon=coeffs_mon
        )

        # Define the integration argument
        def integral_argument_term1_jac(x, coeffs_mon_, k_):

            # First reconstruct the full X matrix
            X_loc = copy.copy(self.X)
            X_loc[:, self.skip_dimensions + k_] = copy.copy(x)

            # Calculate the local basis function matrix
            Psi_mon_loc = self.fun_mon[k_](X_loc, self)

            # Determine the gradients
            rec_arg = np.dot(Psi_mon_loc, coeffs_mon_[:, np.newaxis])[..., 0]

            objective = self.rect.evaluate_dfdc(f=rec_arg, dfdc=Psi_mon_loc)

            return objective

        # Add the integration
        term_1_vector_monotone = self.gauss_quadrature(
            f=integral_argument_term1_jac,
            a=0,
            b=self.X[:, self.skip_dimensions + k],
            args=(coeffs_mon, k),
            **self.quadrature_input,
        )

        # If we have non-monotone terms, consider them
        if self.Psi_nonmon[k] is not None:

            # Evaluate the non-monotone vector term
            term_1_vector_nonmonotone = copy.copy(self.Psi_nonmon[k])

            # Stack the results together
            term_1_vector = np.column_stack(
                (term_1_vector_nonmonotone, term_1_vector_monotone)
            )

        else:

            # If we have no non-monotone terms, the vector is only composed of
            # monotone coefficients
            term_1_vector = term_1_vector_monotone

        # Combine to obtain the full term 1
        term_1 = np.einsum("i,ij->ij", term_1_scalar, term_1_vector)

        # Prepare term 2

        # Create term_2
        rec_arg = np.dot(self.Psi_mon[k], coeffs_mon[:, np.newaxis])[
            ..., 0
        ]  # This is dfdk

        numer = self.rect.evaluate_dfdc(f=rec_arg, dfdc=self.Psi_mon[k])

        denom = 1 / (self.rect.evaluate(rec_arg) + self.delta)

        term_2 = np.einsum("ij,i->ij", numer, denom)

        if div > 0:
            # If we have non-monotone terms, expand the term accordingly
            term_2 = np.column_stack((np.zeros((term_2.shape[0], div)), term_2))

        # Combine both terms
        objective = np.mean(term_1 - term_2, axis=0)

        # Add regularization, if desired

        if self.regularization is not None:

            # A scalar regularization was specified
            if type(self.regularization) == str:

                if self.regularization.lower() == "l1":

                    # Regularization_lambda is identical for all parameters
                    if np.isscalar(self.regularization_lambda):

                        # Add l1 regularization for all coefficients
                        term = np.asarray(
                            list(self.regularization_lambda * np.sign(coeffs_nonmon))
                            + list(self.regularization_lambda * np.sign(coeffs_mon))
                        )

                    elif type(self.regularization_lambda) == list:

                        # Add l1 regularization for all coefficients
                        term = (
                            np.asarray(
                                list(np.sign(coeffs_nonmon)) + list(np.sign(coeffs_mon))
                            )
                            * self.regularization_lambda[k]
                        )

                    else:

                        raise ValueError(
                            "Data type of regularization_lambda not understood. Must be either scalar or list."
                        )

                    objective += term

                elif self.regularization.lower() == "l2":

                    # Regularization_lambda is identical for all parameters
                    if np.isscalar(self.regularization_lambda):

                        # Add l2 regularization for all coefficients
                        term = np.asarray(
                            list(self.regularization_lambda * 2 * coeffs_nonmon)
                            + list(self.regularization_lambda * 2 * coeffs_mon)
                        )

                    elif type(self.regularization_lambda) == list:

                        # Add l2 regularization for all coefficients
                        term = (
                            np.asarray(list(2 * coeffs_nonmon) + list(2 * coeffs_mon))
                            * self.regularization_lambda[k]
                        )

                    else:

                        raise ValueError(
                            "Data type of regularization_lambda not understood. Must be either scalar or list."
                        )

                    objective += term

                else:

                    raise ValueError("regularization_type must be either 'l1' or 'l2'.")

            else:

                raise ValueError(
                    "The variable 'regularization' must be either None, 'l1', or 'l2'."
                )

        return objective

    def inverse_map(self, Y: np.array, X_precalc: np.array = None):

        """This function evaluates the inverse transport map, mapping samples
        from a multivariate standard Gaussian back to the target distribution.
        If X_precalc is specified, the map instead evaluates a conditional of
        the target distribution given X_precalc. The function assumes any
        precalculated output are the FIRST dimensions of the total output.
        If X_precalc is specified, its dimensions and the input dimensions must
        sum to the full dimensionality of sample space.

        :param Y: N-by-D or N-by-(D-E) array of reference distribution samples to be mapped to the target distribution,
            where N is the number of samples, D is the number of target distribution dimensions, and E the number of
            pre-specified dimensions (if X_precalc is specified).
        :param X_precalc: N-by-E array of samples in the space of the target distribution, used to condition the lower
             D-E dimensions during the inversion process.
        :return: N-by-D array of samples in the space of the target distribution.
        """

        # # Check consistency with precalculated output, if applicable if X_precalc is not None: D1  = Y.shape[-1] D2
        # = X_precalc.shape[-1] if D1+D2 != self.D: raise Exception('Dimensions of input ('+str(D1)+') and
        # pre-calculated output ('+str(D2)+') do not sum to dimensions of sample space ('+str(self.D)+').' ) N1  =
        # Y.shape[:-1] N2  = X_precalc.shape[:-1] if N1 != N2: raise Exception('Sample size of input ('+str(N1)+')
        # and pre-calculated output ('+str(N2)+') are not consistent.')

        # Extract number of samples
        N = Y.shape[0]

        # No X_precalc was provided

        if X_precalc is None:  # Yes

            # Initialize the output ensemble
            X = np.zeros((N, self.skip_dimensions + self.D))

            # Go through all dimensions
            for k in np.arange(0, self.D, 1):
                X = self.vectorized_root_search_bisection(Yk=Y[:, k], X=X, k=k)
                # replaces infs and -infs by the mean of the corresponding dimension without the infs and -infs
                X[np.isinf(X[:, k])] = np.mean(X[np.isreal(X[:, k]), k])
                X[np.isneginf(X[:, k])] = np.mean(X[np.isreal(X[:, k]), k])

            # If we standardized the samples, undo the standardization
            if self.standardize_samples:
                X = self.scaler.inverse_transform(X)  # overflow can occur here

        # X_precalc was provided, and matches the reduced map definition

        if X_precalc is not None:

            if X_precalc.shape[-1] == self.skip_dimensions:  # Yes

                # Initialize the output ensemble
                X = np.zeros((N, self.skip_dimensions + self.D))

                # If we standardize the samples, we must also standardize the
                # precalculated values first
                X[:, : self.skip_dimensions] = copy.copy(X_precalc)

                if self.standardize_samples:
                    X[:, : self.skip_dimensions] -= self.scaler.mean_[
                        : self.skip_dimensions
                    ]
                    X[:, : self.skip_dimensions] /= self.scaler.scale_[
                        : self.skip_dimensions
                    ]

                # Go through all dimensions
                for k in np.arange(0, self.D, 1):
                    X = self.vectorized_root_search_bisection(Yk=Y[:, k], X=X, k=k)
                    # replaces infs and -infs by the mean of the corresponding dimension without the infs and -infs
                    X[np.isinf(X[:, k])] = np.mean(X[np.isreal(X[:, k]), k])
                    X[np.isneginf(X[:, k])] = np.mean(X[np.isreal(X[:, k]), k])

                # If we standardized the samples, undo the standardization
                if self.standardize_samples:
                    X = self.scaler.inverse_transform(X)

            # A full map was defined, but so were precalculated values

            elif self.skip_dimensions == 0 and X_precalc is not None:

                # Create a local copy of skip_dimensions
                skip_dimensions = X_precalc.shape[-1]
                D = skip_dimensions + Y.shape[-1]

                # Initialize the output ensemble
                X = np.zeros((N, D))

                # If we standardize the samples, we must also standardize the
                # precalculated values first
                X[:, :skip_dimensions] = X_precalc

                if self.standardize_samples:
                    # Standardize the precalculated samples for the map
                    X[:, :skip_dimensions] -= self.scaler.mean_[:skip_dimensions]
                    X[:, :skip_dimensions] /= self.scaler.scale_[:skip_dimensions]

                # Go through all dimensions
                for i, k in enumerate(np.arange(skip_dimensions, D, 1)):
                    X = self.vectorized_root_search_bisection(Yk=Y[:, i], X=X, k=k)
                    # replaces infs and -infs by the mean of the corresponding dimension without the infs and -infs
                    X[np.isinf(X[:, k])] = np.mean(X[np.isreal(X[:, k]), k])
                    X[np.isneginf(X[:, k])] = np.mean(X[np.isreal(X[:, k]), k])

                # If we standardized the samples, undo the standardization
                if self.standardize_samples:
                    X = self.scaler.inverse_transform(X)

        return X[:, self.skip_dimensions :]

    def vectorized_root_search_bisection(
        self,
        X: np.array,
        Yk: np.array,
        k: int,
    ):

        """This function searches for the roots of the k-th map component
        through bisection. It is called in the inverse_map function.

        :param X: N-by-k array of samples inverted so far, where the k-th column still contains the reference samples
            used as a residual in the root finding process
        :param Yk: a vector containing the target values in the k-th dimension, for which the root finding algorithm
            must solve.
        :param k: an integer variable defining what map component is being evaluated. Corresponds to a dimension of
            sample space.
        """
        # Function to optimize
        def f(x):
            # x is a multivariate vector
            return self.s(x=np.array(list(zip(X[:, 0], x))), k=0) - Yk

        sol = root(f, X[:, 0])  # find the root of the function
        X[:, self.skip_dimensions + k] = sol.x  # store the root in the output array

        return X

    def gauss_quadrature(
        self,
        f,
        a,
        b,
        order=100,
        args=None,
        Ws=None,
        xis=None,
        adaptive=False,
        threshold=1e-6,
        increment=1,
        verbose=False,
        full_output=False,
    ):

        """This function implements a Gaussian quadrature numerical integration
        scheme. It is used if the monotonicity = 'integrated rectifier', for
        which monotonicity is ensured by integrating a strictly positive
        function obtained from a rectifier.

        :param f: function to be integrated element-wise.
        :param a: lower bound for integration, defined as either a scalar or a vector.
        :param b: upper bound for integration, defined as either a scalar or a vector.
        :param order: order of the Legendre polynomial used for the integration scheme.
        :param args: a dictionary with supporting keyword arguments to be passed to the function.
        :param Ws: weights of the integration points, can be calculated in advance to speed up the computation.
            Is calculated by the integration scheme, if not specified.
        :param xis: positions of the integration points, can be calculated in advance to speed up the computation.
             Is calculated by the integration scheme, if not specified.
        :param full_output: Flag for whether the positions and weights of the integration points should be returned
             along with the integration results. If True, returns a tuple with (results,order,xis,Ws). If False, only
             returns results.
        :param adaptive: flag which determines whether the numerical scheme should adjust the order of the Legendre
             polynomial adaptively (True) or use the integer provided by 'order' (False).
        :param threshold: threshold for the difference in the adaptive integration, adaptation stops after difference in
             integration
        :param increment: increment by which the order is increased in each adaptation cycle. Higher values correspond
            to larger steps.
        :param verbose: flag which determines whether information about the integration process should be printed to
             console (True) or not (False).
        :return: the result of the integration, either as a scalar or a vector.
        """

        # Here the actual magic starts

        # If adaptation is desired, we must iterate; prepare a flag for this
        repeat = True
        iteration = 0

        # Iterate, if adaptation = True; Otherwise, iteration stops after one round
        while repeat:

            # Increment the iteration counter
            iteration += 1

            # If required, determine the weights and positions of the integration
            # points; always required if adaptation is active
            if Ws is None or xis is None or adaptive is True:
                # Weights and integration points are not specified; calculate them
                # To get the weights and positions of the integration points, we must
                # provide the *order*-th Legendre polynomial and its derivative
                # As a first step, get the coefficients of both functions
                coefs = np.zeros(order + 1)
                coefs[-1] = 1
                coefs_der = np.polynomial.legendre.legder(coefs)

                # With the coefficients defined, define the Legendre function
                LegendreDer = np.polynomial.legendre.Legendre(coefs_der)

                # Obtain the locations of the integration points
                xis = np.polynomial.legendre.legroots(coefs)

                # Calculate the weights of the integration points
                Ws = 2.0 / ((1.0 - xis ** 2) * (LegendreDer(xis) ** 2))

            # If any of the boundaries is a vector, vectorize the operation
            if not np.isscalar(a) or not np.isscalar(b):

                # If only one of the bounds is a scalar, vectorize it
                if np.isscalar(a) and not np.isscalar(b):
                    a = np.ones(b.shape) * a
                if np.isscalar(b) and not np.isscalar(a):
                    b = np.ones(a.shape) * b

                # Alternative approach, more amenable to dimension-sensitivity in
                # the function f. To speed up computation, pre-calculate the limit
                # differences and sum
                lim_dif = b - a
                lim_sum = b + a

                # To understand what's happening here, consider the following:
                # lim_dif and lim_sum   - shape (N)
                # funcres               - shape (N) up to shape (N-by-C-by-C)

                # If no additional arguments were given, simply call the function
                if args is None:

                    result = (
                        lim_dif
                        * 0.5
                        * (Ws[0] * f(lim_dif * 0.5 * xis[0] + lim_sum * 0.5))
                    )

                    for i in np.arange(1, len(Ws)):
                        result += (
                            lim_dif
                            * 0.5
                            * (Ws[i] * f(lim_dif * 0.5 * xis[i] + lim_sum * 0.5))
                        )

                # Otherwise, pass the arguments on as well
                else:

                    funcres = f(lim_dif * 0.5 * xis[0] + lim_sum * 0.5, *args)

                    # Depending on what shape the output function returns, we
                    # must take special precautions to ensure the product works

                    # If the function output is the same size as its input
                    if len(funcres.shape) == len(lim_dif.shape):

                        result = lim_dif * 0.5 * (Ws[0] * funcres)

                        for i in np.arange(1, len(Ws)):
                            funcres = f(lim_dif * 0.5 * xis[i] + lim_sum * 0.5, *args)

                            result += lim_dif * 0.5 * (Ws[i] * funcres)

                    # If the function output has one dimension more than its
                    # corresponding input
                    elif len(funcres.shape) == len(lim_dif.shape) + 1:

                        result = np.einsum("i,ij->ij", lim_dif * 0.5 * Ws[0], funcres)

                        for i in np.arange(1, len(Ws)):
                            funcres = f(lim_dif * 0.5 * xis[i] + lim_sum * 0.5, *args)

                            result += np.einsum(
                                "i,ij->ij", lim_dif * 0.5 * Ws[i], funcres
                            )

                    # If the function output has one dimension more than its
                    # corresponding input
                    elif len(funcres.shape) == len(lim_dif.shape) + 2:

                        result = np.einsum("i,ijk->ijk", lim_dif * 0.5 * Ws[0], funcres)

                        for i in np.arange(1, len(Ws)):
                            funcres = f(lim_dif * 0.5 * xis[i] + lim_sum * 0.5, *args)

                            result += np.einsum(
                                "i,ijk->ijk", lim_dif * 0.5 * Ws[i], funcres
                            )

                    else:
                        raise Exception(
                            f"Shape of input dimension is {lim_sum.shape} and shape of output dimension is {funcres.shape}. Currently, we have only implemented situations in which input and output are the same shape, or where output is one or two dimensions larger. "
                        )

            else:

                # Now start the actual computation.

                # If no additional arguments were given, simply call the function
                if args is None:
                    result = (
                        (b - a)
                        * 0.5
                        * np.sum(Ws * f((b - a) * 0.5 * xis + (b + a) * 0.5))
                    )
                # Otherwise, pass the arguments on as well
                else:
                    result = (
                        (b - a)
                        * 0.5
                        * np.sum(Ws * f((b - a) * 0.5 * xis + (b + a) * 0.5, *args))
                    )

            # if adaptive, store results for next iteration
            if adaptive:

                # In the first iteration, just store the results
                if iteration == 1:
                    previous_result = copy.copy(result)

                # In later iterations, check integration process
                else:

                    # How much did the results change?
                    change = np.abs(result - previous_result)

                    # Check if the change in results was sufficient
                    if np.max(change) < threshold or iteration > 1000:

                        # Stop iterating
                        repeat = False

                        if iteration > 1000 and self.verbose:
                            logger.warning(
                                f"Adaptive integration stopped after 1000 iteration cycles. Final change: {change}"
                            )

                        # Print the final change if required
                        if verbose and self.verbose:
                            logger.info(
                                f"Final maximum change of Gauss Quadrature: {np.max(change)}"
                            )

                # If we must still continue repeating, increment order and store
                # current result for next iteration
                if repeat:
                    order += increment
                    previous_result = copy.copy(result)

            # If no adaptation is required, simply stop iterating
            else:
                repeat = False

        # If full output is desired
        if full_output:
            result = (result, order, xis, Ws)

        if verbose and self.verbose:
            logger.info(f"Order: {order}")

        return result

    class Rectifier:
        def __init__(self, mode: str = "softplus", delta: float = 1e-8):

            """This object specifies what function is used to rectify the monotone
            map component functions if monotonicity = 'integrated rectifier',
            before the rectifier's output is integrated to yield a monotone
            map component in x_k.

            :param mode: keyword string defining which function is used to rectify the map component functions.
            :param delta: a small offset value to prevent arithmetic underflow in some rectifier functions.
            :return: result: the rectified function value.
            """

            self.mode = mode
            self.delta = delta

        def evaluate(self, X: np.array) -> np.array:

            """This function evaluates the specified rectifier.

            :param X: an array of function evaluates to be rectified.
            """

            if self.mode == "squared":

                res = X ** 2

            elif self.mode == "exponential":

                res = np.exp(X)

            elif self.mode == "expneg":

                res = np.exp(-X)

            elif self.mode == "softplus":

                a = np.log(2)
                aX = a * X
                below = aX < 0
                aX[below] = 0
                res = np.log(1 + np.exp(-np.abs(a * X))) + aX

            elif self.mode == "explinearunit":

                res = np.zeros(X.shape)
                res[(X < 0)] = np.exp(X[(X < 0)])
                res[(X >= 0)] = X[(X >= 0)] + 1

            return res

        def inverse(self, X):

            """This function evaluates the inverse of the specified rectifier.

            :param X: an array of function evaluates to be rectified.
            :return: result: the inverse of the rectified function value.
            """

            if len(np.where(X < 0)[0] > 0):
                raise Exception("Input to inverse rectifier are negative.")

            if self.mode == "squared":

                raise Exception("Squared rectifier is not invertible.")

            elif self.mode == "exponential":

                res = np.log(X)

            elif self.mode == "expneg":

                res = -np.log(X)

            elif self.mode == "softplus":

                a = np.log(2)

                opt1 = np.log(np.exp(a * X) - 1)
                opt2 = X

                opt1idx = opt1 - opt2 >= 0
                opt2idx = opt1 - opt2 < 0

                res = np.zeros(X.shape)
                res[opt1idx] = opt1[opt1idx]
                res[opt2idx] = opt2[opt2idx]

            elif self.mode == "explinearunit":

                res = np.zeros(X.shape)

                below = X < 1
                above = X >= 1

                res[below] = np.log(X[below])
                res[above] = X - 1

            return res

        def evaluate_dx(self, X: np.array) -> np.array:

            """This function evaluates the derivative of the specified
            rectifier.

            :param X: an array of function evaluates to be rectified.
            :return: result: the derivative of the rectified function value.
            """

            if self.mode == "squared":

                res = 2 * X

            elif self.mode == "exponential":

                res = np.exp(X)

            elif self.mode == "expneg":

                res = -np.exp(-X)

            elif self.mode == "softplus":

                a = np.log(2)
                res = 1 / (1 + np.exp(-a * X))

            elif self.mode == "explinearunit":

                below = X < 0
                above = X >= 0

                res = np.zeros(X.shape)

                res[below] = np.exp(X[below])
                res[above] = 0

            return res

        def evaluate_dfdc(self, f, dfdc):

            """This function evaluates terms used in the optimization of the
            map components if monotonicity = 'separable monotonicity'.

            :param f: an array of function evaluates to be rectified.
            :param dfdc: an array of function evaluates to be rectified.
            :return: result: the derivative of the rectified function value.
            """

            if self.mode == "squared":

                raise Exception("Not implemented yet.")

            elif self.mode == "exponential":

                res = np.exp(f)

                # Combine with dfdc
                res = np.einsum("i,ij->ij", res, dfdc)

            elif self.mode == "expneg":

                res = -np.exp(-f)

                # Combine with dfdc
                res = np.einsum("i,ij->ij", res, dfdc)

            elif self.mode == "softplus":

                # Calculate the first part
                a = np.log(2)
                res = 1 / (1 + np.exp(-a * f))

                # Combine with dfdc
                res = np.einsum("i,ij->ij", res, dfdc)

            elif self.mode == "explinearunit":

                raise Exception("Not implemented yet.")

            return res

        def logevaluate(self, X: np.array) -> np.array:

            """This function evaluates the logarithm of the specified
            rectifier.

            :param X: an array of function evaluates to be rectified.
            :return: result: the logarithm of the rectified function value.
            """

            if self.mode == "squared":

                res = np.log(X ** 2)

            elif self.mode == "exponential":

                if self.delta == 0:
                    res = X
                else:
                    res = np.log(np.exp(X) + self.delta)

            elif self.mode == "expneg":

                res = -X

            elif self.mode == "softplus":

                a = np.log(2)
                aX = a * X
                below = aX < 0
                aX[below] = 0
                res = np.log(1 + np.exp(-np.abs(a * X))) + aX

                res = np.log(res + self.delta)

            elif self.mode == "explinearunit":

                res = np.zeros(X.shape)
                res[(X < 0)] = np.exp(X[(X < 0)])
                res[(X >= 0)] = X[(X >= 0)] + 1

                res = np.log(res)

            return res
