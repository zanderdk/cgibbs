#include "prob.h"

char stringprob_docs[] = "Calculates the probability of MC m generating string s.";

PyMethodDef prob_funcs[] = {
	{	"probs",
		(PyCFunction)probs,
		METH_VARARGS,
		stringprob_docs
		},
	{NULL}
};

char prob_mod_docs[] = "Calculate probability of MCs.";

PyModuleDef prob_mod = {
	PyModuleDef_HEAD_INIT,
	"prob",
	prob_mod_docs,
	-1,
	prob_funcs,
	NULL,
	NULL,
	NULL,
	NULL
};

PyMODINIT_FUNC PyInit_prob(void) {
	return PyModule_Create(&prob_mod);
}
