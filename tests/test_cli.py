import pytest
from click.testing import CliRunner
from unittest.mock import patch, MagicMock

from transitkit.cli import main


class TestCLI:
    """Test command line interface."""
    
    @pytest.fixture
    def runner(self):
        """Create CLI runner."""
        return CliRunner()
    
    def test_version(self, runner):
        """Test version command."""
        result = runner.invoke(main, ["--version"])
        assert result.exit_code == 0
        assert "transitkit" in result.output.lower()
    
    def test_help(self, runner):
        """Test help command."""
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "Usage:" in result.output
    
    def test_load_command(self, runner):
        """Test load command."""
        with patch('transitkit.cli.load_target') as mock_load:
            mock_load.return_value = MagicMock()
            
            result = runner.invoke(main, [
                "load", "TIC 123456", "--mission", "TESS", "--sector", "25"
            ])
            
            assert result.exit_code == 0
            mock_load.assert_called_once_with(
                "TIC 123456",
                mission="TESS",
                sector=25
            )
    
    def test_fit_command(self, runner):
        """Test fit command."""
        with patch('transitkit.cli.fit_transit') as mock_fit:
            mock_result = MagicMock()
            mock_result.parameters = {"period": 10.0}
            mock_fit.return_value = mock_result
            
            result = runner.invoke(main, [
                "fit", "test_data.csv", "--method", "mcmc", "--output", "results.json"
            ])
            
            assert result.exit_code == 0
            mock_fit.assert_called_once()
    
    @pytest.mark.integration
    def test_batch_command_integration(self, runner, temp_dir):
        """Test batch command integration."""
        # Create test targets file
        targets_file = temp_dir / "targets.txt"
        targets_file.write_text("TIC 123456\nTIC 789012\n")
        
        # Create output directory
        output_dir = temp_dir / "results"
        
        with patch('transitkit.cli.BatchProcessor') as MockProcessor:
            mock_processor = MagicMock()
            mock_processor.process.return_value = []
            MockProcessor.return_value = mock_processor
            
            result = runner.invoke(main, [
                "batch", str(targets_file),
                "--output-dir", str(output_dir),
                "--workers", "2"
            ])
            
            assert result.exit_code == 0
            MockProcessor.assert_called_once_with(
                output_dir=str(output_dir),
                n_workers=2,
                overwrite=False
            )