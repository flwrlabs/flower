// Copyright 2024 Flower Labs GmbH. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ==============================================================================
export function warnPreviewFeature(name: string) {
  console.warn(
    `PREVIEW FEATURE: ${name}

            This is a preview feature.It could change significantly or be removed
            entirely in future versions of Flower.
        `,
  )
}

export function warnDeprecatedFeature(name: string) {
  console.warn(
    `DEPRECATED FEATURE: ${name}

            This is a deprecated feature.It will be removed
            entirely in future versions of Flower.
        `,
  )
}

export function warnDeprecatedFeatureWithExample(deprecation_message: string, example_message: string, code_example: string) {
  console.warn(
    `DEPRECATED FEATURE: ${deprecation_message}

            Check the following \`FEATURE UPDATE\` warning message for the preferred
            new mechanism to use this feature in Flower.
        `,
  )
  console.warn(
    `FEATURE UPDATE: ${example_message}
        ------------------------------------------------------------
        ${code_example}
        ------------------------------------------------------------
       `,
  )
}

export function warnUnsupportedFeature(name: string) {
  console.warn(
    `UNSUPPORTED FEATURE: ${name}

            This is an unsupported feature.It will be removed
            entirely in future versions of Flower.
        `,
  )
}
