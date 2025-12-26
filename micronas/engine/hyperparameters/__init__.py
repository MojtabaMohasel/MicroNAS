# Copyright 2019 The KerasTuner Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from micronas import utils
from micronas.api_export import keras_tuner_export
from micronas.backend import keras
from micronas.engine.hyperparameters import hp_types
from micronas.engine.hyperparameters.hp_types import Boolean
from micronas.engine.hyperparameters.hp_types import Choice
from micronas.engine.hyperparameters.hp_types import Fixed
from micronas.engine.hyperparameters.hp_types import Float
from micronas.engine.hyperparameters.hp_types import Int
from micronas.engine.hyperparameters.hyperparameter import HyperParameter
from micronas.engine.hyperparameters.hyperparameters import HyperParameters

OBJECTS = hp_types.OBJECTS + (
    HyperParameter,
    HyperParameters,
)

ALL_CLASSES = {cls.__name__: cls for cls in OBJECTS}


@keras_tuner_export("micronas.engine.hyperparameters.deserialize")
def deserialize(config):
    return utils.deserialize_keras_object(config, module_objects=ALL_CLASSES)


@keras_tuner_export("micronas.engine.hyperparameters.serialize")
def serialize(obj):
    return utils.serialize_keras_object(obj)
