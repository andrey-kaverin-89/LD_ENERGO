import Page from "../components/Page";

// import Box from '@mui/material/Box';
// import List from '@material-ui/core/List';
// import ListItem from '@material-ui/core/ListItem';
// import ListItemButton from '@material-ui/core/ListItemButton';
// import ListItemIcon from '@material-ui/core/ListItemIcon';
// import ListItemText from '@material-ui/core/ListItemText';
// import Divider from '@material-ui/core';
import { Box } from "@material-ui/core";
// import InboxIcon from '@mui/icons-material/Inbox';
// import DraftsIcon from '@mui/icons-material/Drafts';

export default function Regions() {
    return (
        <Page>
            <h1>Regions</h1>

            <Box sx={{ width: '100%', maxWidth: 360, bgcolor: 'background.paper' }}>
                {/* asdasd
                
                    <List>
                        <ListItem disablePadding>
                            <ListItemButton>
                            <ListItemIcon>
                                <InboxIcon />
                            </ListItemIcon>
                            <ListItemText primary="Inbox" />
                            </ListItemButton>
                        </ListItem>
                        <ListItem disablePadding>
                            <ListItemButton>
                            <ListItemIcon>
                                <DraftsIcon />
                            </ListItemIcon>
                            <ListItemText primary="Drafts" />
                            </ListItemButton>
                        </ListItem>
                    </List>
                 */}
                {/* <Divider /> */}
                {/* <nav aria-label="secondary mailbox folders">
                    <List>
                    <ListItem disablePadding>
                        <ListItemButton>
                        <ListItemText primary="Trash" />
                        </ListItemButton>
                    </ListItem>
                    <ListItem disablePadding>
                        <ListItemButton component="a" href="#simple-list">
                        <ListItemText primary="Spam" />
                        </ListItemButton>
                    </ListItem>
                    </List>
                </nav> */}
                </Box>
        </Page>
    )
}