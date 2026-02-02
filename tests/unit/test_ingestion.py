"""Unit tests for data ingestion module."""

import pytest
import numpy as np
from pathlib import Path
import tempfile

from habitat_pipeline.ingestion import DataLoader, MetadataParser
from habitat_pipeline.ingestion.loader import BaseDataLoader
from habitat_pipeline.ingestion.formats import BinaryLoader


class TestMetadataParser:
    """Test metadata parser functionality."""
    
    def test_init(self):
        """Test metadata parser initialization."""
        parser = MetadataParser()
        assert parser.metadata == {}
    
    def test_get_set(self):
        """Test getting and setting metadata values."""
        parser = MetadataParser()
        parser.set('key', 'value')
        assert parser.get('key') == 'value'
        assert parser.get('nonexistent', 'default') == 'default'
    
    def test_update(self):
        """Test updating metadata."""
        parser = MetadataParser()
        parser.update({'key1': 'value1', 'key2': 'value2'})
        assert parser.get('key1') == 'value1'
        assert parser.get('key2') == 'value2'
    
    def test_validate(self):
        """Test metadata validation."""
        parser = MetadataParser()
        parser.update({'name': 'test', 'age': 5})
        
        schema = {'name': str, 'age': int}
        assert parser.validate(schema) is True
        
        schema_invalid = {'name': str, 'missing': int}
        assert parser.validate(schema_invalid) is False


class TestBinaryLoader:
    """Test binary data loader."""
    
    def test_binary_loader_init(self):
        """Test binary loader initialization."""
        # Create temporary binary file
        with tempfile.NamedTemporaryFile(suffix='.bin', delete=False) as f:
            temp_file = Path(f.name)
            
            # Write some data
            n_channels = 4
            n_samples = 1000
            data = np.random.randint(-1000, 1000, size=(n_samples, n_channels), dtype=np.int16)
            data.tofile(f)
        
        try:
            # Load data
            loader = BinaryLoader(
                str(temp_file),
                num_channels=n_channels,
                sampling_rate=30000.0,
                dtype=np.int16
            )
            
            assert loader.num_channels == n_channels
            assert loader.sampling_rate == 30000.0
            
            # Load data
            loaded_data, fs = loader.load_ephys()
            assert loaded_data.shape[0] == n_channels
            assert loaded_data.shape[1] == n_samples
            assert fs == 30000.0
            
        finally:
            # Cleanup
            temp_file.unlink()
    
    def test_binary_loader_metadata(self):
        """Test binary loader metadata."""
        with tempfile.NamedTemporaryFile(suffix='.bin', delete=False) as f:
            temp_file = Path(f.name)
            data = np.random.randint(-1000, 1000, size=(1000, 4), dtype=np.int16)
            data.tofile(f)
        
        try:
            loader = BinaryLoader(
                str(temp_file),
                num_channels=4,
                sampling_rate=30000.0
            )
            
            metadata = loader.load_metadata()
            assert metadata['num_channels'] == 4
            assert metadata['sampling_rate'] == 30000.0
            assert metadata['format'] == 'Binary'
            
        finally:
            temp_file.unlink()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
