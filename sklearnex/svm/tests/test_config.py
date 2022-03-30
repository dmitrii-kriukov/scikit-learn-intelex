#!/usr/bin/env python
#===============================================================================
# Copyright 2021 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#===============================================================================

import numpy as np
from numpy.testing import assert_allclose

def test_bf_default():
    from sklearnex.svm import SVC
    from sklearnex import config_context
    X = np.array([[-2, -1], [-1, -1], [-1, -2],
                  [+1, +1], [+1, +2], [+2, +1]])
    y = np.array([1, 1, 1, 2, 2, 2])
    svc = SVC(kernel='linear').fit(X, y)
    assert_allclose(svc.dual_coef_, [[-0.25, .25]])

def test_bf_balanced():
    from sklearnex.svm import SVC
    from sklearnex import patch_sklearn
    patch_sklearn()
    from sklearn import config_context
    X = np.array([[-2, -1], [-1, -1], [-1, -2],
                  [+1, +1], [+1, +2], [+2, +1]])
    y = np.array([1, 1, 1, 2, 2, 2])
    with config_context(fp32_with_bf16_emulation='balanced'):
        svc = SVC(kernel='linear').fit(X, y)
        assert_allclose(svc.dual_coef_, [[-0.25, .25]])