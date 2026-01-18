# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_signal


from sila2.framework import Feature
from sila2.server import FeatureImplementationBase, SilaServer


class GenericFeatureImplementation(FeatureImplementationBase):  # type: ignore[misc]
    """
    A generic implementation for dynamically loaded SiLA features.
    """

    def __init__(self, parent_server: SilaServer, feature_name: str) -> None:
        super().__init__(parent_server)
        self.feature_name = feature_name


def generate_minimal_feature_xml(feature_name: str) -> str:
    """
    Generates a valid, minimal SiLA Feature Definition XML.
    """
    return f"""<?xml version="1.0" encoding="utf-8" ?>
<Feature SiLA2Version="1.0" FeatureVersion="1.0" MaturityLevel="Draft" Originator="com.coreason" Category="Dynamic"
         xmlns="http://www.sila-standard.org"
         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
         xsi:schemaLocation="http://www.sila-standard.org https://gitlab.com/SiLA2/sila_base/raw/master/schema/FeatureDefinition.xsd">
    <Identifier>{feature_name}</Identifier>
    <DisplayName>{feature_name}</DisplayName>
    <Description>Dynamically generated capability for {feature_name}</Description>
</Feature>"""


class FeatureRegistry:
    """
    Registry to manage dynamic feature loading.
    """

    @staticmethod
    def create_feature(feature_name: str) -> Feature:
        """
        Create a SiLA Feature object from a name.
        """
        xml_def = generate_minimal_feature_xml(feature_name)
        return Feature(feature_definition=xml_def)

    @staticmethod
    def create_implementation(server: SilaServer, feature_name: str) -> FeatureImplementationBase:
        """
        Create a default implementation for the feature.
        """
        return GenericFeatureImplementation(server, feature_name)
