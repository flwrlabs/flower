"""
Usage: python dev/build-docker-image-matrix.py --flwr-version <flower version e.g. 1.13.0>

Images are built in three workflows: stable, nightly, and unstable (main).
Each builds for `amd64` and `arm64`.

1. **Ubuntu Images**:
   - Used for images where dependencies might be installed by users.
   - Ubuntu uses `glibc`, compatible with most ML frameworks.

2. **Alpine Images**:
   - Used only for minimal images (e.g., SuperLink) where no extra dependencies are expected.
   - Limited use due to dependency (in particular ML frameworks) compilation complexity with `musl`.

Workflow Details:
- **Stable Release**: Triggered on new releases. Builds full matrix (all Python versions, Ubuntu and Alpine).
- **Nightly Release**: Daily trigger. Builds full matrix (latest Python, Ubuntu only).
- **Unstable**: Triggered on main branch commits. Builds simplified matrix (latest Python, Ubuntu only).
"""

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

# when we switch to Python 3.11 in the ci, we need to change the DistroName to:
# class DistroName(StrEnum):
#     ALPINE = "alpine"
#     UBUNTU = "ubuntu"
assert sys.version_info < (3, 11), "Script requires Python 3.9 or lower."


class DistroName(str, Enum):
    ALPINE = "alpine"
    UBUNTU = "ubuntu"


@dataclass
class Distro:
    name: "DistroName"
    version: str


LATEST_SUPPORTED_PYTHON_VERSION = "3.14"
SUPPORTED_PYTHON_VERSIONS = [
    "3.10",
    "3.11",
    "3.12",
    "3.13",
    LATEST_SUPPORTED_PYTHON_VERSION,
]

DOCKERFILE_ROOT = "framework/docker"


@dataclass
class Variant:
    distro: Distro
    extras: Optional[Any] = None


@dataclass
class CpuVariant:
    pass


@dataclass
class CudaVariant:
    version: str


CUDA_VERSIONS_CONFIG = [
    ("11.2.2", "20.04"),
    ("11.8.0", "22.04"),
    ("12.1.0", "22.04"),
    ("12.3.2", "22.04"),
]
LATEST_SUPPORTED_CUDA_VERSION = Variant(
    Distro(DistroName.UBUNTU, "22.04"),
    CudaVariant(version="12.4.1"),
)

# ubuntu base image
UBUNTU_VARIANT = Variant(
    Distro(DistroName.UBUNTU, "24.04"),
    CpuVariant(),
)


# alpine base image
ALPINE_VARIANT = Variant(
    Distro(DistroName.ALPINE, "3.22"),
    CpuVariant(),
)


# ubuntu cuda base images
CUDA_VARIANTS = [
    Variant(
        Distro(DistroName.UBUNTU, ubuntu_version),
        CudaVariant(version=cuda_version),
    )
    for (cuda_version, ubuntu_version) in CUDA_VERSIONS_CONFIG
] + [LATEST_SUPPORTED_CUDA_VERSION]


def remove_patch_version(version: str) -> str:
    return ".".join(version.split(".")[0:2])


@dataclass
class BaseImageBuilder:
    file_dir_fn: Callable[[Any], str]
    tags_fn: Callable[[Any], list[str]]
    build_args_fn: Callable[[Any], str]
    build_args: Any
    tags: list[str] = field(init=False)
    file_dir: str = field(init=False)
    tags_encoded: str = field(init=False)
    build_args_encoded: str = field(init=False)


@dataclass
class BaseImage(BaseImageBuilder):
    namespace_repository: str = "flwr/base"

    @property
    def file_dir(self) -> str:
        return self.file_dir_fn(self.build_args)

    @property
    def tags(self) -> str:
        return self.tags_fn(self.build_args)

    @property
    def tags_encoded(self) -> str:
        return "\n".join(self.tags)

    @property
    def build_args_encoded(self) -> str:
        return self.build_args_fn(self.build_args)


@dataclass
class BinaryImage:
    namespace_repository: str
    file_dir: str
    base_image: str
    tags_encoded: str


def new_binary_image(
    name: str,
    base_image: BaseImage,
    tags_fn: Optional[Callable],
) -> Dict[str, Any]:
    tags = []
    if tags_fn is not None:
        tags += tags_fn(base_image) or []

    return BinaryImage(
        f"flwr/{name}",
        f"{DOCKERFILE_ROOT}/{name}",
        base_image.tags[0],
        "\n".join(tags),
    )


def generate_binary_images(
    name: str,
    base_images: List[BaseImage],
    tags_fn: Optional[Callable] = None,
    filter: Optional[Callable] = None,
) -> List[Dict[str, Any]]:
    filter = filter or (lambda _: True)

    return [
        new_binary_image(name, image, tags_fn) for image in base_images if filter(image)
    ]


def tag_superlink_supernode_images(image: BaseImage) -> List[str]:
    """
    Compute the Docker image tags based on its build arguments.

    - If the image is built on Alpine with the latest supported Python version,
      append the Flower framework version to the existing tags.
    - Else if the image is built on Ubuntu with the latest supported Python version
      and a CPU-only variant, append the "latest" tag to the existing tags.
    - Otherwise, return the original tags unchanged.
    """
    if (
        image.build_args.variant.distro.name == DistroName.ALPINE
        and image.build_args.python_version == LATEST_SUPPORTED_PYTHON_VERSION
    ):
        return image.tags + [image.build_args.flwr_version]
    elif (
        image.build_args.variant.distro.name == DistroName.UBUNTU
        and image.build_args.python_version == LATEST_SUPPORTED_PYTHON_VERSION
        and isinstance(image.build_args.variant.extras, CpuVariant)
    ):
        return image.tags + ["latest"]
    else:
        return image.tags


def tag_superexec_images(image: BaseImage) -> List[str]:
    """
    Compute the Docker image tags based on its build arguments.

    For images built on Ubuntu with the latest supported Python version
    and a CPU variant, this will append the Flower framework version
    and the "latest" tag to the existing tags list. All other images
    simply retain their original tags.
    """
    if (
        image.build_args.variant.distro.name == DistroName.UBUNTU
        and image.build_args.python_version == LATEST_SUPPORTED_PYTHON_VERSION
        and isinstance(image.build_args.variant.extras, CpuVariant)
    ):
        return image.tags + [image.build_args.flwr_version, "latest"]
    else:
        return image.tags


#
# Build matrix for stable releases
#
def build_stable_matrix(flwr_version: str) -> List[BaseImage]:
    @dataclass
    class StableBaseImageBuildArgs:
        variant: Variant
        python_version: str
        flwr_version: str

    cpu_build_args = """PYTHON_VERSION={python_version}
FLWR_VERSION={flwr_version}
DISTRO={distro_name}
DISTRO_VERSION={distro_version}
"""

    cpu_build_args_variants = [
        StableBaseImageBuildArgs(UBUNTU_VARIANT, python_version, flwr_version)
        for python_version in SUPPORTED_PYTHON_VERSIONS
    ] + [
        StableBaseImageBuildArgs(
            ALPINE_VARIANT, LATEST_SUPPORTED_PYTHON_VERSION, flwr_version
        )
    ]

    cpu_base_images = [
        BaseImage(
            file_dir_fn=lambda args: f"{DOCKERFILE_ROOT}/base/{args.variant.distro.name.value}",
            tags_fn=lambda args: [
                f"{args.flwr_version}-py{args.python_version}-{args.variant.distro.name.value}{args.variant.distro.version}"
            ],
            build_args_fn=lambda args: cpu_build_args.format(
                python_version=args.python_version,
                flwr_version=args.flwr_version,
                distro_name=args.variant.distro.name,
                distro_version=args.variant.distro.version,
            ),
            build_args=build_args_variant,
        )
        for build_args_variant in cpu_build_args_variants
    ]

    cuda_build_args_variants = [
        StableBaseImageBuildArgs(variant, python_version, flwr_version)
        for variant in CUDA_VARIANTS
        for python_version in SUPPORTED_PYTHON_VERSIONS
    ]

    cuda_build_args = cpu_build_args + """CUDA_VERSION={cuda_version}"""

    cuda_base_image = [
        BaseImage(
            file_dir_fn=lambda args: f"{DOCKERFILE_ROOT}/base/{args.variant.distro.name.value}-cuda",
            tags_fn=lambda args: [
                f"{args.flwr_version}-py{args.python_version}-cu{remove_patch_version(args.variant.extras.version)}-{args.variant.distro.name.value}{args.variant.distro.version}",
            ],
            build_args_fn=lambda args: cuda_build_args.format(
                python_version=args.python_version,
                flwr_version=args.flwr_version,
                distro_name=args.variant.distro.name,
                distro_version=args.variant.distro.version,
                cuda_version=args.variant.extras.version,
            ),
            build_args=build_args_variant,
        )
        for build_args_variant in cuda_build_args_variants
    ]

    # base_images = cpu_base_images + cuda_base_image
    base_images = cpu_base_images

    binary_images = (
        # ubuntu and alpine images for the latest supported python version
        generate_binary_images(
            "superlink",
            base_images,
            tag_superlink_supernode_images,
            lambda image: image.build_args.python_version
            == LATEST_SUPPORTED_PYTHON_VERSION
            and isinstance(image.build_args.variant.extras, CpuVariant),
        )
        # ubuntu images for each supported python version
        + generate_binary_images(
            "supernode",
            base_images,
            tag_superlink_supernode_images,
            lambda image: (
                image.build_args.variant.distro.name == DistroName.UBUNTU
                and isinstance(image.build_args.variant.extras, CpuVariant)
            )
            or (
                image.build_args.variant.distro.name == DistroName.ALPINE
                and image.build_args.python_version == LATEST_SUPPORTED_PYTHON_VERSION
            ),
        )
        # ubuntu images for each supported python version
        + generate_binary_images(
            "superexec",
            base_images,
            tag_superexec_images,
            lambda image: image.build_args.variant.distro.name == DistroName.UBUNTU,
        )
    )

    return base_images, binary_images


#
# Build matrix for unstable releases
#
def build_unstable_matrix(flwr_version_ref: str) -> List[BaseImage]:
    @dataclass
    class UnstableBaseImageBuildArgs:
        variant: Variant
        python_version: str
        flwr_version_ref: str

    cpu_ubuntu_build_args_variant = UnstableBaseImageBuildArgs(
        UBUNTU_VARIANT, LATEST_SUPPORTED_PYTHON_VERSION, flwr_version_ref
    )

    cpu_build_args = """PYTHON_VERSION={python_version}
FLWR_VERSION_REF={flwr_version_ref}
DISTRO={distro_name}
DISTRO_VERSION={distro_version}
"""

    cpu_base_image = BaseImage(
        file_dir_fn=lambda args: f"{DOCKERFILE_ROOT}/base/{args.variant.distro.name.value}",
        tags_fn=lambda _: ["unstable"],
        build_args_fn=lambda args: cpu_build_args.format(
            python_version=args.python_version,
            flwr_version_ref=args.flwr_version_ref,
            distro_name=args.variant.distro.name,
            distro_version=args.variant.distro.version,
        ),
        build_args=cpu_ubuntu_build_args_variant,
    )

    cuda_build_args_variant = UnstableBaseImageBuildArgs(
        LATEST_SUPPORTED_CUDA_VERSION, LATEST_SUPPORTED_PYTHON_VERSION, flwr_version_ref
    )

    cuda_build_args = cpu_build_args + """CUDA_VERSION={cuda_version}"""

    cuda_base_image = BaseImage(
        file_dir_fn=lambda args: f"{DOCKERFILE_ROOT}/base/{args.variant.distro.name.value}-cuda",
        tags_fn=lambda _: ["unstable-cuda"],
        build_args_fn=lambda args: cuda_build_args.format(
            python_version=args.python_version,
            flwr_version_ref=args.flwr_version_ref,
            distro_name=args.variant.distro.name,
            distro_version=args.variant.distro.version,
            cuda_version=args.variant.extras.version,
        ),
        build_args=cuda_build_args_variant,
    )

    # base_images = [cpu_base_image, cuda_base_image]
    base_images = [cpu_base_image]

    binary_images = (
        generate_binary_images(
            "superlink",
            base_images,
            lambda image: image.tags,
            lambda image: isinstance(image.build_args.variant.extras, CpuVariant),
        )
        + generate_binary_images(
            "supernode",
            base_images,
            lambda image: image.tags,
            lambda image: isinstance(image.build_args.variant.extras, CpuVariant),
        )
        + generate_binary_images("superexec", base_images, lambda image: image.tags)
    )

    return base_images, binary_images


#
# Build matrix for nightly releases
#
def build_nightly_matrix(flwr_version: str, flwr_package: str) -> List[BaseImage]:
    @dataclass
    class NightlyBaseImageBuildArgs:
        variant: Variant
        python_version: str
        flwr_version: str
        flwr_package: str

    cpu_ubuntu_build_args_variant = NightlyBaseImageBuildArgs(
        UBUNTU_VARIANT, LATEST_SUPPORTED_PYTHON_VERSION, flwr_version, flwr_package
    )

    cpu_build_args = """PYTHON_VERSION={python_version}
FLWR_VERSION={flwr_version}
FLWR_PACKAGE={flwr_package}
DISTRO={distro_name}
DISTRO_VERSION={distro_version}
"""

    cpu_base_image = BaseImage(
        file_dir_fn=lambda args: f"{DOCKERFILE_ROOT}/base/{args.variant.distro.name.value}",
        tags_fn=lambda args: [args.flwr_version, "nightly"],
        build_args_fn=lambda args: cpu_build_args.format(
            python_version=args.python_version,
            flwr_version=args.flwr_version,
            flwr_package=args.flwr_package,
            distro_name=args.variant.distro.name,
            distro_version=args.variant.distro.version,
        ),
        build_args=cpu_ubuntu_build_args_variant,
    )

    cuda_build_args_variant = NightlyBaseImageBuildArgs(
        LATEST_SUPPORTED_CUDA_VERSION,
        LATEST_SUPPORTED_PYTHON_VERSION,
        flwr_version,
        flwr_package,
    )

    cuda_build_args = cpu_build_args + """CUDA_VERSION={cuda_version}"""

    cuda_base_image = BaseImage(
        file_dir_fn=lambda args: f"{DOCKERFILE_ROOT}/base/{args.variant.distro.name.value}-cuda",
        tags_fn=lambda args: [f"{args.flwr_version}-cuda", "nightly-cuda"],
        build_args_fn=lambda args: cuda_build_args.format(
            python_version=args.python_version,
            flwr_version=args.flwr_version,
            flwr_package=args.flwr_package,
            distro_name=args.variant.distro.name,
            distro_version=args.variant.distro.version,
            cuda_version=args.variant.extras.version,
        ),
        build_args=cuda_build_args_variant,
    )

    # base_images = [cpu_base_image, cuda_base_image]
    base_images = [cpu_base_image]

    binary_images = (
        generate_binary_images(
            "superlink",
            base_images,
            lambda image: image.tags,
            lambda image: isinstance(image.build_args.variant.extras, CpuVariant),
        )
        + generate_binary_images(
            "supernode",
            base_images,
            lambda image: image.tags,
            lambda image: isinstance(image.build_args.variant.extras, CpuVariant),
        )
        + generate_binary_images("superexec", base_images, lambda image: image.tags)
    )

    return base_images, binary_images


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(
        description="Generate Github Docker workflow matrix"
    )
    arg_parser.add_argument("--flwr-version", type=str, required=True)
    arg_parser.add_argument("--flwr-package", type=str, default="flwr")
    arg_parser.add_argument(
        "--matrix", choices=["stable", "nightly", "unstable"], default="stable"
    )

    args = arg_parser.parse_args()

    flwr_version = args.flwr_version
    flwr_package = args.flwr_package
    matrix = args.matrix

    if matrix == "stable":
        base_images, binary_images = build_stable_matrix(flwr_version)
    elif matrix == "nightly":
        base_images, binary_images = build_nightly_matrix(flwr_version, flwr_package)
    else:
        base_images, binary_images = build_unstable_matrix(flwr_version)

    print(
        json.dumps(
            {
                "base": {
                    "images": list(
                        map(
                            lambda image: asdict(
                                image,
                                dict_factory=lambda x: {
                                    k: v
                                    for (k, v) in x
                                    if v is not None and callable(v) is False
                                },
                            ),
                            base_images,
                        )
                    )
                },
                "binary": {
                    "images": list(map(lambda image: asdict(image), binary_images))
                },
            }
        )
    )
