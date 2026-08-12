"""
Usage: python dev/build-docker-image-matrix.py \
    (--flwr-version VERSION | --flwr-version-ref REF) [--image-tag TAG]

Images are built for `amd64` and `arm64` from a versioned package or Git ref.

1. **Ubuntu Images**:
   - Used for images where dependencies might be installed by users.
   - Ubuntu uses `glibc`, compatible with most ML frameworks.

2. **Alpine Images**:
   - Used only for minimal images (e.g., SuperLink) where no extra dependencies are expected.
   - Limited use due to dependency (in particular ML frameworks) compilation complexity with `musl`.

Every caller builds the same full matrix across all supported Python versions,
Ubuntu, and Alpine.
"""

import argparse
import json
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any, Callable, Dict, List, Optional


class DistroName(StrEnum):
    ALPINE = "alpine"
    UBUNTU = "ubuntu"


@dataclass
class Distro:
    name: "DistroName"
    version: str


LATEST_SUPPORTED_PYTHON_VERSION = "3.13"
SUPPORTED_PYTHON_VERSIONS = [
    "3.11",
    "3.12",
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


def build_matrix(
    flwr_version: str,
    flwr_version_ref: str,
    flwr_package: str,
    image_tags: list[str],
) -> List[BaseImage]:
    """Build the complete Framework image matrix."""

    @dataclass
    class BaseImageBuildArgs:
        variant: Variant
        python_version: str
        flwr_version: str
        flwr_version_ref: str
        flwr_package: str
        image_tags: list[str]

    cpu_build_args = """PYTHON_VERSION={python_version}
FLWR_VERSION={flwr_version}
FLWR_VERSION_REF={flwr_version_ref}
FLWR_PACKAGE={flwr_package}
DISTRO={distro_name}
DISTRO_VERSION={distro_version}
"""

    cpu_build_args_variants = [
        BaseImageBuildArgs(
            UBUNTU_VARIANT,
            python_version,
            flwr_version,
            flwr_version_ref,
            flwr_package,
            image_tags,
        )
        for python_version in SUPPORTED_PYTHON_VERSIONS
    ] + [
        BaseImageBuildArgs(
            ALPINE_VARIANT,
            LATEST_SUPPORTED_PYTHON_VERSION,
            flwr_version,
            flwr_version_ref,
            flwr_package,
            image_tags,
        )
    ]

    def tags(args: BaseImageBuildArgs) -> list[str]:
        variant_tags = [
            f"{tag}-py{args.python_version}-{args.variant.distro.name.value}{args.variant.distro.version}"
            for tag in args.image_tags
        ]
        if (
            args.variant == UBUNTU_VARIANT
            and args.python_version == LATEST_SUPPORTED_PYTHON_VERSION
        ):
            return variant_tags + args.image_tags
        return variant_tags

    cpu_base_images = [
        BaseImage(
            file_dir_fn=lambda args: f"{DOCKERFILE_ROOT}/base/{args.variant.distro.name.value}",
            tags_fn=tags,
            build_args_fn=lambda args: cpu_build_args.format(
                python_version=args.python_version,
                flwr_version=args.flwr_version,
                flwr_version_ref=args.flwr_version_ref,
                flwr_package=args.flwr_package,
                distro_name=args.variant.distro.name,
                distro_version=args.variant.distro.version,
            ),
            build_args=build_args_variant,
        )
        for build_args_variant in cpu_build_args_variants
    ]

    cuda_build_args_variants = [
        BaseImageBuildArgs(
            variant,
            python_version,
            flwr_version,
            flwr_version_ref,
            flwr_package,
            image_tags,
        )
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
                flwr_version_ref=args.flwr_version_ref,
                flwr_package=args.flwr_package,
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
            lambda image: image.tags,
            lambda image: image.build_args.python_version
            == LATEST_SUPPORTED_PYTHON_VERSION
            and isinstance(image.build_args.variant.extras, CpuVariant),
        )
        # ubuntu images for each supported python version
        + generate_binary_images(
            "supernode",
            base_images,
            lambda image: image.tags,
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
            lambda image: image.tags,
            lambda image: image.build_args.variant.distro.name == DistroName.UBUNTU,
        )
    )

    return base_images, binary_images


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(
        description="Generate Github Docker workflow matrix"
    )
    source = arg_parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--flwr-version", type=str)
    source.add_argument("--flwr-version-ref", type=str)
    arg_parser.add_argument("--flwr-package", type=str, default="flwr")
    arg_parser.add_argument("--image-tag", action="append")
    # Retain the stable-only argument until framework-release.yml is migrated.
    arg_parser.add_argument("--matrix", choices=["stable"], help=argparse.SUPPRESS)

    args = arg_parser.parse_args()
    image_tags = args.image_tag or ([args.flwr_version] if args.flwr_version else None)
    if image_tags is None:
        arg_parser.error("--image-tag is required with --flwr-version-ref")

    base_images, binary_images = build_matrix(
        flwr_version=args.flwr_version or "",
        flwr_version_ref=args.flwr_version_ref or "",
        flwr_package=args.flwr_package,
        image_tags=image_tags,
    )

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
