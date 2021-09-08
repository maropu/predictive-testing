#!/usr/bin/env python3

#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from pydantic import BaseModel, Field
from typing import List

"""
Type validation classes for Github REST APIs
"""


class RateLimit(BaseModel):
    limit: int = Field(ge=0)
    remaining: int = Field(ge=0)
    reset: int = Field(ge=0)
    used: int = Field(ge=0)


class ResourceLimit(BaseModel):
    core: RateLimit
    search: RateLimit
    graphql: RateLimit
    graphql: RateLimit


class RateLimits(BaseModel):
    resources: ResourceLimit
    rate: RateLimit


class Author(BaseModel):
    login: str = Field(min_length=1, max_length=39)


class ContributorStat(BaseModel):
    author: Author
    total: int = Field(ge=1)


class ContributorStats(BaseModel):
    stats: List[ContributorStat]
