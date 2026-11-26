# Copyright 2026 Flower Labs GmbH. All Rights Reserved.
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
# ==============================================================================
"""Readable OpenAPI descriptions for binary protobuf messages."""

from google.protobuf.descriptor import FieldDescriptor
from google.protobuf.descriptor_pb2 import FieldDescriptorProto  # pylint: disable=E0611
from google.protobuf.json_format import MessageToJson
from google.protobuf.message import Message

from flwr.proto.control_pb2_docs import PROTOBUF_DESCRIPTIONS


def protobuf_message_description(example: Message) -> str:
    """Describe a protobuf message using its descriptor and source comments."""
    descriptor = example.DESCRIPTOR
    source_root = "https://github.com/flwrlabs/flower/blob/main/framework/proto/"
    lines = [
        f"Binary protobuf message: `{descriptor.full_name}`.",
        f"[Protobuf definition]({source_root}{descriptor.file.name}).",
        "",
    ]
    if summary := PROTOBUF_DESCRIPTIONS.get(descriptor.full_name):
        lines.extend([summary, ""])
    for field in descriptor.fields:
        field_type = FieldDescriptorProto.Type.Name(field.type).removeprefix("TYPE_")
        type_name = field_type.lower()
        if field.message_type is not None:
            type_name = (
                f"[{field.message_type.full_name}]"
                f"({source_root}{field.message_type.file.name})"
            )
        elif field.enum_type is not None:
            type_name = (
                f"[{field.enum_type.full_name}]"
                f"({source_root}{field.enum_type.file.name})"
            )
        presence = "implicit"
        if field.label == FieldDescriptor.LABEL_REPEATED:
            presence = "repeated"
        elif field.has_presence:
            presence = "explicit"
        note = PROTOBUF_DESCRIPTIONS.get(field.full_name, "")
        lines.append(
            f"- **`{field.name}`** ({type_name}, field {field.number}, "
            f"{presence} presence): {note}"
        )
    lines.extend(
        [
            "",
            "**Decoded example (JSON for readability, not the HTTP wire format).** "
            "Default-valued fields may be omitted. Send and receive serialized "
            "protobuf bytes with `Content-Type: application/protobuf`.",
            "",
            "```json",
            MessageToJson(example, preserving_proto_field_name=True).strip(),
            "```",
        ]
    )
    return "\n".join(lines)
