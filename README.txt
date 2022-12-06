Hi Professor!

The code that I worked with in development was a disorganized mess which allowed
me to complete the assignment, but under no circumstance should be read by a human being.
Accordingly, I have prepared a clean version which allows you to compute the model
according to the final architecture proposed in the paper (including the removal of
faces before computing influence scores, which I flagged as a possible improvement, but
not computed for this submission), as well as datafiles which allow you to more easily
reproduce the results of the paper (every step of the pipeline has been made runnable by
a human being, using the data procured from the paper). Follow these instructions to
reproduce:

pipeline_1.py:

This file will produce the models which will be used to compute the influence weights.
The file is setup to produce a NONFACE models, as described in the architecture section
of the paper. The models which is used in the experiments section is stored in files
such as "model_age__smoteless_35.h5". These models will be used downstream in the default
setup. This pipeline will produce models of the form "model_age.h5" which do not contain
face images, and can be used in future pipelines to compute the architecture described
in the paper. NOTE: This file uses a preprocessed set of images as "faces" and "not
faces". To reprocess, change the variable "REPROCESS" to True.

influence.py:

This file will produce influence scores for the images specified at the header of the
file (such as AGE_IMAGE = "TRAIN0300.png"). To run this with the final architecture,
it is necessary to examine the files produced as an output of pipeline_1 and choose
an image which it outputted for which you (as the model runner) can correctly identify
as belonging to the given class. The results of this step trained against the legacy
model (non-faces included) are stored in files of the form "uploss_scores_age_legacy.dill",
but the actual files produced by running this step will not include the word "legacy".
As a final note, this step will use the legacy model to learn the influence, USE_PIPELINE_1
is set to True; and this will not work until pipeline_1.py is finished.

pipeline_3.py:

This file generates the final models for the architecture, and has been equipped with extra
configuration to generate the data from the paper. The configuration parameters work as follows:

PERCENT - The percent of high influence score images to filter
LEGACY - Whether to use the precomputed influences scores, or the scores computed in influence.py
SMOTE - Whether to apply SMOTE to the data
USE_INFLUENCE - Whether to filter influence points at al
FILTER_FACES - Whether to filter to just faces

Based on these configuration parameters, you can produce any results from the paper or by
producing your own values!



NOTE!! When running my code on my personal machine versus the lab machine, I noticed an import
failed on one and not the other and vice versa. So if the import

from keras.utils import load_img, img_to_array, array_to_img

fails on your machine, you can try replacing it with:

from keras.preprocessing.image import load_img, img_to_array, array_to_img