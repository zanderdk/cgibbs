#include <Python.h>
#include <numpy/arrayobject.h>
#include "prob.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

int traceToPath(int lab, int mcSize, double trans[mcSize][mcSize], long labels[mcSize], int s){
    for (int i = 0; i < mcSize; i++) {
        if (trans[s][i] > 0 && labels[i] == lab)
            return i;
    }
    return -1;
}

double calcPathProb(int pathLen, long path[pathLen], int mcSize, double trans[mcSize][mcSize]) {
    double res = 0;
    for(int i = 1; i < pathLen; i++) {
        int s = path[i-1];
        int t = path[i];
        if (s == -1 || t == -1)
            return log(0);
        res += log(trans[s][t]);
    }
    return res;
}

double calcProbDeterministic(int mcSize, int strLen, double trans[mcSize][mcSize], long labels[mcSize], long *str, long s) {
    long path[strLen];
    path[0] = s;

    for (int i = 1; i < strLen; i++){
        int k = traceToPath(str[i], mcSize, trans, labels, path[i-1]);
        path[i] = k;
    }


    return calcPathProb(strLen, path, mcSize, trans);
}

PyObject * probs(PyObject *self, PyObject *args) {

    PyObject *m, *Pystr, *npLabels;
    PyObject *sObj;
    PyArrayObject *npTrans;
    int s;

    if(!PyArg_ParseTuple(args, "OO", &m, &Pystr))
		return NULL;

    long strlen = PyList_Size(Pystr);
	long str[strlen];
	for (int i = 0; i < strlen; i++) {
	    PyObject *obj = PyList_GetItem(Pystr, i);
	    str[i] = PyLong_AsLong(obj);
	}

    npTrans = (PyArrayObject *)PyObject_GetAttrString(m, "transitions");
    npLabels = PyObject_GetAttrString(m, "labels");
    sObj = PyObject_GetAttrString(m, "init");

    s = (int)PyLong_AsLong(sObj);

    int mcSize = npTrans->dimensions[0];

    double trans[mcSize][mcSize];
    long labels[mcSize];

    for (int i = 0; i < mcSize; i++) {
        for(int j = 0; j < mcSize; j++) {
            double *k = PyArray_GETPTR2(npTrans, i, j);
            double f = *k;
            trans[i][j] = (double)f;
        }
    }

    for (int i = 0; i < mcSize; i++) {
        PyObject *obj = PyList_GetItem(npLabels, i);
	    labels[i] = PyLong_AsLong(obj);
    }

    double res = calcProbDeterministic(mcSize, strlen, trans, labels, str, s);
    return PyFloat_FromDouble(res);
}